#if defined(STRATA_HY4_FLASH_MMQ)
#include "mmq/moe_mmq.hpp"
#include <set>
#endif
#include "strata/hy4flash_gguf/expert_heat.hpp"
#include "strata/kernels/native_router.hpp"

#include <nvtx3/nvToolsExt.h>
namespace { struct OffloadNvtx { explicit OffloadNvtx(const char* n){nvtxRangePushA(n);} ~OffloadNvtx(){nvtxRangePop();} }; }
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/dfly.hpp"

#include "strata/hy4flash_gguf/engram_io.hpp"
#include "strata/ngram/ple_reader.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/hy4flash_gguf/host_memory.hpp"
#include "cpu_split_fit.hpp"
#include "strata/core/host_graph_launch.hpp"
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/window.hpp"
#include "strata/hy4flash_gguf/round_state.hpp"
#include "strata/hy4flash_gguf/mtp.hpp"
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
#include "strata/hy4flash_gguf/gpu_sampler.hpp"
#endif
#include "strata/hy4flash_gguf/prefill_moe.hpp"
#include "strata/hy4flash_gguf/cpu_experts.hpp"
#include "strata/hy4flash_gguf/engram.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/ihc.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/hy4flash_gguf/model_file.hpp"
#include "strata/hy4flash_gguf/fp8.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"

#include <algorithm>
#include <array>
#include <new>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <limits>
#include <stdexcept>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::hy4_flash {

// Collect row-read statistics separately from the synchronization used in phase timing.
double g_engram_secs = 0.0;      ///< wall inside the row read
long g_engram_steps = 0;         ///< calls that read rows
long g_engram_requests = 0;      ///< rows asked for, with repeats
long g_engram_recur = 0;         ///< of those, ones whose id had been asked for before in this process
std::unordered_set<uint64_t> g_engram_seen;   ///< STRATA_HY4_FLASH_ENGRAM_STATS=1 only

/// The recurrence set is unbounded, so it is opt-in: 660,006,416 possible rows and a long run would grow
/// it without limit. The timing above is always on, because two `steady_clock` reads a step are free.
bool engram_stats_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ENGRAM_STATS");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void engram_read_cost(double& secs, long& steps, long& requests, long& recur, long& distinct) {
    secs = g_engram_secs;
    steps = g_engram_steps;
    requests = g_engram_requests;
    recur = g_engram_recur;
    distinct = (long) g_engram_seen.size();
}

namespace {

/// Set from ForwardOptions for the duration of one call; read by Buf.
bool g_poison = false;

struct Buf {
    void* p = nullptr;
    bool owns = true;
    static Buf borrowed(void* p) { Buf b; b.p=p; b.owns=false; return b; }
    Buf() = default;
    /// Fill fresh scratch with NaNs when checking for uninitialized reads.
    explicit Buf(std::size_t bytes) {
        if (bytes != 0 && cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
        if (p == nullptr) return;
        if (!g_poison) {
            cudaMemset(p, 0, bytes);
            return;
        }
        // 0x7fc00000 is a quiet NaN as float32; written byte-wise so it covers any element type.
        static const std::vector<uint8_t> pat = [] {
            const uint32_t nan_bits = 0x7fc00000u;
            std::vector<uint8_t> v(4u << 20);
            for (std::size_t i = 0; i + 4 <= v.size(); i += 4) std::memcpy(v.data() + i, &nan_bits, 4);
            return v;
        }();
        std::size_t at = 0;
        while (at < bytes) {
            const std::size_t n = std::min(pat.size(), bytes - at);
            cudaMemcpy(static_cast<char*>(p) + at, pat.data(), n, cudaMemcpyHostToDevice);
            at += n;
        }
    }
    ~Buf() { reset(); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p(o.p), owns(o.owns) { o.p = nullptr; o.owns=true; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            reset();
            p = o.p; owns = o.owns;
            o.p = nullptr; o.owns = true;
        }
        return *this;
    }
    void reset() {
        if (p && owns) cudaFree(p);
        p = nullptr; owns = true;
    }
    /// Hand the allocation to someone else - the residency, which outlives this pass.
    void* release() {
        void* q = p;
        p = nullptr;
        return q;
    }
    template <class T>
    T* as() const { return static_cast<T*>(p); }
};

struct LayerMajorCall {
    int layer=0,loaded=-1;float* state=nullptr;
    Buf weights,identity;std::size_t weight_capacity=0;int identity_count=0;
    bool stream_experts = false;
    // Own the super-chunk residual outside reusable/borrowed scratch. It stays live across every
    // layer and is released by the outer call before decode can start.
    Buf state_storage;
    bool state_device=false;
    std::size_t state_bytes=0,state_offset=0;
    ExpertGroup group{};
    std::size_t h2d=0,d2d=0;
};
thread_local LayerMajorCall* g_layer_major=nullptr;
bool layer_major_state_device(){
    static const bool on=[]{const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_LM_R");
        return v && std::string(v)=="vram";}();
    return on;
}
bool prefill_borrow_on(){static const bool on=[]{const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_BORROW");return v&&std::atoi(v)!=0;}();return on;}
thread_local bool g_auto_dispatch=false;
thread_local int g_auto_chunk=0;
bool auto_chunk_on(){
    static const bool on=[] {
        const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_CHUNK");
        return v && std::strcmp(v,"auto")==0;
    }();return on;
}
int auto_chunk_cap(){
    static const int cap=[] {
        const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_AUTO_MAX");
        return v?std::max(128,std::min(8192,std::atoi(v))):8192;
    }();return cap;
}

bool layer_major_on(){static const bool on=[]{const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR");return v&&std::atoi(v)!=0;}();return on;}

/// A tensor of the GGUF, uploaded to the device. Reads in chunks so a 1 GB expert tensor never needs a 1 GB
/// host buffer of its own.
struct DenseArenaError : std::runtime_error { using std::runtime_error::runtime_error; };

class Gguf {
public:
    /// Keep the read-only GGUF mapping alive for all tensor views into it.
    explicit Gguf(const std::string& path) : meta_(path) {}
    ~Gguf() {
        // DenseResidency tears down Work, workers and captured graphs before its Open/Gguf.
        if (dense_arena_.p != nullptr) {
            const cudaError_t rc = cudaFree(dense_arena_.release());
            std::fprintf(stderr, "HY4_FLASH_DENSE_ARENA free bytes=%zu rc=%d\n", dense_arena_bytes_, (int) rc);
        }
        // DenseResidency destroys its Work (and joins the copy/CPU workers) before this mapping.
        for (const auto& range : registered_expert_ranges_)
            (void) cudaHostUnregister(reinterpret_cast<void*>(range.first));
    }
    bool ok() const { return true; } // ModelFile construction either maps all shards or throws.
    bool mapped() const { return true; }

    /// A stable pointer to `bytes` at `file_offset`, or nullptr when there is no mapping or it would run off
    /// the end. Stable is the point: a device copy may read it after this returns.
    const void* at(uint64_t file_offset, std::size_t bytes) const {
        return meta_.at(file_offset, bytes);
    }

    // Opt-in decode supply: pin only expert tensor pages, not the Engram/dense portions of the
    // GGUF. Coalesce page-rounded ranges before registration: neighbouring tensors may share a
    // boundary page. Preserve original quantized bytes and the GPU's three-plane layout.
    bool pin_expert_ranges(const std::vector<std::pair<uint64_t, std::size_t>>& tensors) {
        expert_pin_attempted_ = true;
#ifndef _WIN32
        const long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0 || tensors.empty()) return false;
        const uintptr_t mask = static_cast<uintptr_t>(page - 1);
        if ((static_cast<uintptr_t>(page) & mask) != 0) return false;
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        for (const auto& tensor : tensors) {
            const void* p = at(tensor.first, tensor.second);
            if (!p || !tensor.second) return false;
            const uintptr_t begin = reinterpret_cast<uintptr_t>(p);
            const uintptr_t end = begin + tensor.second;
            if (end < begin || end + mask < end) return false;
            ranges.emplace_back(begin & ~mask, (end + mask) & ~mask);
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;
        for (const auto& range : ranges) {
            if (!merged.empty() && range.first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, range.second);
            else merged.push_back(range);
        }
        // Register large host ranges in chunks to bound the kernel's page-registration allocations.
        constexpr std::size_t max_register_bytes = 32u << 20;
        std::vector<std::pair<uintptr_t, uintptr_t>> chunks;
        for (const auto& [begin, end] : merged) {
            for (uintptr_t pos = begin; pos < end;) {
                const uintptr_t next = pos + std::min<uintptr_t>(end - pos, max_register_bytes);
                chunks.emplace_back(pos, next);
                pos = next;
            }
        }
        if (!registered_expert_ranges_.empty()) return registered_expert_ranges_ == chunks;
        registered_expert_ranges_.reserve(chunks.size());
        const auto start = std::chrono::steady_clock::now();
        std::size_t bytes = 0;
        for (const auto& range : chunks) {
            const auto status = cudaHostRegister(reinterpret_cast<void*>(range.first),
                range.second - range.first, cudaHostRegisterReadOnly | cudaHostRegisterPortable);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "HY4_FLASH_PIN_EXPERT_MMAP failed: %s; bytes=%zu registered=%zu; falling back to staged copies\n",
                             cudaGetErrorString(status), std::size_t(range.second - range.first), registered_expert_ranges_.size());
                (void) cudaGetLastError();
                for (const auto& live : registered_expert_ranges_)
                    (void) cudaHostUnregister(reinterpret_cast<void*>(live.first));
                registered_expert_ranges_.clear();
                (void) cudaGetLastError();
                return false;
            }
            registered_expert_ranges_.push_back(range);
            bytes += range.second - range.first;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::fprintf(stderr, "HY4_FLASH_PIN_EXPERT_MMAP registered ranges=%zu bytes=%zu ms=%.3f readonly=1\n",
            registered_expert_ranges_.size(), bytes, ms);
        return true;
#else
        (void) tensors;
        return false;
#endif
    }
    bool expert_pin_attempted() const { return expert_pin_attempted_; }
    bool expert_pin_ready() const { return !registered_expert_ranges_.empty(); }

    // CUDA validates the source extent against one registration per copy.
    size_t expert_copy_span(const void* src, size_t bytes) const {
        const auto addr = reinterpret_cast<uintptr_t>(src);
        const auto it = std::upper_bound(registered_expert_ranges_.begin(), registered_expert_ranges_.end(),
            addr, [](uintptr_t p, const auto& range) { return p < range.second; });
        if (it == registered_expert_ranges_.end()) return bytes;
        const auto boundary = addr < it->first ? it->first : it->second;
        return std::min<size_t>(bytes, boundary - addr);
    }

    cudaError_t copy_expert_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) const {
        auto* to = static_cast<char*>(dst);
        const auto* from = static_cast<const char*>(src);
        while (bytes) {
            const auto n = expert_copy_span(from, bytes);
            const auto rc = cudaMemcpyAsync(to, from, n, cudaMemcpyHostToDevice, stream);
            if (rc != cudaSuccess) return rc;
            to += n; from += n; bytes -= n;
        }
        return cudaSuccess;
    }

    cudaError_t copy_expert_rows_async(char* dst, size_t pitch, const char* src,
                                      size_t width, size_t rows, cudaStream_t stream) const {
        if (pitch == width) return copy_expert_async(dst, src, width * rows, stream);
        while (rows) {
            const size_t whole = expert_copy_span(src, width * rows) / width;
            const size_t n = std::max<size_t>(1, whole);
            const auto rc = whole ? cudaMemcpy2DAsync(dst, pitch, src, width, width, n, cudaMemcpyHostToDevice, stream)
                                  : copy_expert_async(dst, src, width, stream);
            if (rc != cudaSuccess) return rc;
            dst += pitch * n; src += width * n; rows -= n;
        }
        return cudaSuccess;
    }

    const strata::TensorInfo* info(const std::string& name) const { return meta_.info(name); }

    // Default-off owner for the exact native tensor bytes, never a dequantized copy.
    bool dense_arena_ready() const { return dense_arena_.p != nullptr; }
    bool dense_arena_failed() const { return dense_arena_failed_; }
    void dense_take_check(const std::string& name) {
        static const int fail_at = [] {
            const char* v=std::getenv("STRATA_HY4_FLASH_DENSE_ARENA_FAIL_TAKE");
            return v ? std::atoi(v) : 0;
        }();
        if (fail_at > 0 && name == "blk.20.ffn_routed_up.weight" && ++dense_take_count_ == fail_at) {
            dense_arena_failed_=true;
            throw DenseArenaError("dense arena injected take failure");
        }
    }
    bool prepare_dense_arena(const std::vector<std::string>& names, std::string& err) {
        if (dense_arena_ready()) return true;
        const auto start = std::chrono::steady_clock::now();
        std::unordered_map<std::string, std::size_t> offsets;
        std::unordered_map<std::string, std::size_t> guards;
        std::size_t total = 0, payload = 0, guard_total = 0;
        for (const auto& name : names) {
            if (offsets.find(name) != offsets.end()) continue;
            const TensorInfo* t = info(name);
            if (!t) continue; // Preserve the existing absent-name negative cache.
            const uint64_t raw = tensor_payload_bytes(*t);
            const std::size_t max = std::numeric_limits<std::size_t>::max();
            if (raw > max || total > max - 255 || payload > max - (std::size_t) raw) {
                err = "dense arena size overflow"; return false;
            }
            total = (total + 255) & ~(std::size_t) 255;
            if ((std::size_t) raw > max - total) { err = "dense arena size overflow"; return false; }
            offsets.emplace(name, total); total += (std::size_t) raw; payload += (std::size_t) raw;
            // MMQ may load the final partial K tile beyond the last tensor row. Its padded
            // activation is zero, but a neighbouring tensor's NaN scale is NOT safe (NaN*0).
            // Match the existing MMQ allocator contract for every supported tensor/type.
            std::size_t guard = 0;
#if defined(STRATA_HY4_FLASH_MMQ)
            if (!t->shape.empty() && t->shape[0] <= (uint64_t)std::numeric_limits<int64_t>::max())
                guard = strata::hy4_flash::mmq::overread_bytes((int)t->type,(int64_t)t->shape[0]);
#endif
            if (guard > max-total || guard > max-guard_total) {
                err="dense arena MMQ guard overflow";return false;
            }
            guards.emplace(name,guard);total+=guard;guard_total+=guard;
        }
        if (total == 0) { err = "dense arena: no tensors"; return false; }
        if (total > std::numeric_limits<std::size_t>::max() - 255) {
            err = "dense arena final alignment overflow"; return false;
        }
        total = (total + 255) & ~(std::size_t) 255;
        std::size_t before=0, device_total=0, after=0;
        cudaMemGetInfo(&before, &device_total);
        Buf candidate(total);
        if (!candidate.p) { err = "dense arena allocation failed: " + std::to_string(total); return false; }
        cudaMemGetInfo(&after, &device_total);
        const char* inject = std::getenv("STRATA_HY4_FLASH_DENSE_ARENA_FAIL_STAGE");
        const int fail_stage = inject ? std::atoi(inject) : 0;
        auto fail = [&](const char* why) {
            err = why;
            const cudaError_t rc = cudaFree(candidate.release());
            std::fprintf(stderr, "HY4_FLASH_DENSE_ARENA rollback bytes=%zu rc=%d\n", total, (int)rc);
            return false;
        };
        if (fail_stage == 1) return fail("dense arena injected failure after allocation");
        std::vector<uint8_t> read_buffer;
        std::size_t copied = 0;
        for (const auto& name : names) {
            auto it = offsets.find(name);
            if (it == offsets.end()) continue;
            // The input list is unique; duplicate names were removed by the caller.
            const TensorInfo* t = info(name);
            const std::size_t bytes = (std::size_t) tensor_payload_bytes(*t);
            const uint64_t file_offset = data_start() + t->offset;
            if (file_offset < data_start()) return fail("dense arena GGUF offset overflow");
            auto* dst = static_cast<uint8_t*>(candidate.p) + it->second;
            // Preserve the old upload's host source path. A mapped dense tensor can overlap
            // page-rounded registered expert ranges; a large mixed-registration H2D source
            // is not equivalent to the original fread staging buffer.
            if (read_buffer.empty()) read_buffer.resize(16u << 20);
            for (std::size_t off = 0; off < bytes;) {
                const std::size_t n = std::min(read_buffer.size(), bytes - off);
                if (!read_at(file_offset + off, read_buffer.data(), n))
                    return fail("dense arena tensor read failed");
                const cudaError_t rc = cudaMemcpy(dst + off, read_buffer.data(), n, cudaMemcpyHostToDevice);
                if (rc != cudaSuccess) {
                    std::fprintf(stderr,"HY4_FLASH_DENSE_ARENA upload_error tensor=%s offset=%zu bytes=%zu error=%s\n",
                                 name.c_str(),off,n,cudaGetErrorString(rc));
                    return fail("dense arena tensor upload failed");
                }
                off += n;
            }
            const std::size_t guard = guards.at(name);
            if (guard != 0) {
                const cudaError_t rc = cudaMemset(dst+bytes,0,guard);
                if (rc != cudaSuccess) {
                    std::fprintf(stderr,"HY4_FLASH_DENSE_ARENA guard_error tensor=%s bytes=%zu error=%s\n",
                                 name.c_str(),guard,cudaGetErrorString(rc));
                    return fail("dense arena MMQ guard initialization failed");
                }
                std::fprintf(stderr,"HY4_FLASH_DENSE_ARENA_GUARD tensor=%s bytes=%zu\n",name.c_str(),guard);
            }
            if (++copied == 1 && fail_stage == 2) return fail("dense arena injected failure after first tensor");
        }
        if (fail_stage == 3) return fail("dense arena injected failure before publication");
        // Publish only after every synchronous upload succeeded. No Slot yet contains an arena pointer.
        dense_offsets_.swap(offsets);
        dense_arena_bytes_ = total;
        dense_arena_ = std::move(candidate);
        const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::fprintf(stderr, "HY4_FLASH_DENSE_ARENA tensors=%zu payload=%zu allocated=%zu guard_bytes=%zu physical_delta=%zu init_ms=%.3f\n",
                     dense_offsets_.size(), payload, total, guard_total, before >= after ? before-after : 0, ms);
        return true;
    }
    void* dense_pointer(const std::string& name) const {
        const auto it = dense_offsets_.find(name);
        if (it == dense_offsets_.end()) {
            if (info(name)) {
                dense_arena_failed_=true;
                throw DenseArenaError("dense arena enumeration missed existing tensor: " + name);
            }
            return nullptr;
        }
        return static_cast<uint8_t*>(dense_arena_.p) + it->second;
    }

    /// Upload `name`; an empty Buf when it is absent or does not fit.
    Buf upload(const std::string& name, int* type = nullptr, std::vector<int64_t>* shape = nullptr) {
        const strata::TensorInfo* t = info(name);
        if (t == nullptr) return Buf();
        const std::size_t bytes = (std::size_t) strata::tensor_payload_bytes(*t);
        Buf b(bytes);
        if (b.p == nullptr) return Buf();
        if (chunk_.empty()) chunk_.resize(16u << 20);
        std::size_t at = 0;
        while (at < bytes) {
            const std::size_t want = std::min(chunk_.size(), bytes - at);
            if (!read_at(data_start() + t->offset + at, chunk_.data(), want)) return Buf();
            if (cudaMemcpy(static_cast<char*>(b.p) + at, chunk_.data(), want, cudaMemcpyHostToDevice) !=
                cudaSuccess)
                return Buf();
            at += want;
        }
        if (type) *type = (int) t->type;
        if (shape) {
            shape->clear();
            for (uint64_t d : t->shape) shape->push_back((int64_t) d);
        }
        return b;
    }

    /// Raw bytes at an offset, usable from several threads at once - which `read_at` is not, because the
    /// seek and the read are two calls on one shared descriptor.
    bool pread_at(uint64_t file_offset, void* dst, std::size_t bytes) const {
        return meta_.pread_at(file_offset, dst, bytes);
    }

    /// Raw bytes of a slice, for the Engram table's rows.
    bool read_at(uint64_t file_offset, void* dst, std::size_t bytes) {
        return meta_.read_at(file_offset, dst, bytes);
    }
    uint64_t data_start() const { return meta_.data_start(); }
    bool physical(uint64_t offset, std::size_t bytes, std::string& path, uint64_t& file_offset) const {
        return meta_.physical(offset, bytes, path, file_offset);
    }

private:
    mutable bool dense_arena_failed_ = false;
    int dense_take_count_ = 0;
    Buf dense_arena_;
    std::size_t dense_arena_bytes_ = 0;
    std::unordered_map<std::string, std::size_t> dense_offsets_;
    ModelFile meta_;
    std::vector<uint8_t> chunk_;
    std::vector<std::pair<uintptr_t, uintptr_t>> registered_expert_ranges_;
    bool expert_pin_attempted_ = false;
};

/// One Q8_0 row of `n` values -> float.
void dequant_q8_0_row(const uint8_t* src, int n, float* out) {
    for (int b = 0; b * 32 < n; ++b) {
        uint16_t h;
        std::memcpy(&h, src + (std::size_t) b * 34, 2);
        const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
        const int exp = (h >> 10) & 0x1f;
        const uint32_t man = h & 0x3ffu;
        float d;
        if (exp == 0) {
            d = (float) man * 5.9604645e-8f;
            if (sign) d = -d;
        } else {
            const uint32_t bits = sign | (uint32_t) ((exp - 15 + 127) << 23) | (man << 13);
            std::memcpy(&d, &bits, 4);
        }
        for (int i = 0; i < 32 && b * 32 + i < n; ++i)
            out[b * 32 + i] = d * (float) (int8_t) src[(std::size_t) b * 34 + 2 + (std::size_t) i];
    }
}


std::string blk(int l) { return "blk." + std::to_string(l) + "."; }

/// STRATA_HY4_FLASH_GRAPHS=0 turns graph capture off, for telling what it is worth and for a card or driver
/// where capture misbehaves. On by default.
bool graphs_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_GRAPHS");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

/// Keep reader threads alive across batches to avoid per-token thread creation.
class Pool {
public:
    explicit Pool(int n) {
        if (n < 1) n = 1;
        for (int i = 0; i < n; ++i) worker_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::unique_lock<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& t : worker_) t.join();
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    /// Run `fn(i)` for i in [0, n), on these threads, and return when all of them have.
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        if (n == 1 || worker_.empty()) {
            fn(0);
            return;
        }
        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = &fn;
            total_ = n;
            next_ = 0;
            done_ = 0;
        }
        cv_.notify_all();
        // This thread takes a share too, so a batch of 2 does not idle the caller.
        for (;;) {
            int i;
            {
                std::unique_lock<std::mutex> lk(m_);
                if (next_ >= total_) break;
                i = next_++;
            }
            fn(i);
            std::unique_lock<std::mutex> lk(m_);
            ++done_;
            if (done_ == total_) cv_done_.notify_all();
        }
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return done_ == total_; });
        fn_ = nullptr;
        total_ = 0;
    }

private:
    void loop() {
        for (;;) {
            int i = -1;
            const std::function<void(int)>* fn = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stop_ || (fn_ != nullptr && next_ < total_); });
                if (stop_) return;
                i = next_++;
                fn = fn_;
            }
            (*fn)(i);
            std::unique_lock<std::mutex> lk(m_);
            ++done_;
            if (done_ == total_) cv_done_.notify_all();
        }
    }
    std::vector<std::thread> worker_;
    std::mutex m_;
    std::condition_variable cv_, cv_done_;
    const std::function<void(int)>* fn_ = nullptr;
    int total_ = 0, next_ = 0, done_ = 0;
    bool stop_ = false;
};

/// Leave enough device memory for forward scratch when admitting weights to the cache.
bool room_for(std::size_t bytes) {
    const std::size_t headroom = (std::size_t) 5 << 30;
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    return free_b > bytes + headroom;
}

/// Read only the selected expert's gate, up and down matrices.
struct ExpertSlices {
    /// Allocate pinned staging per position so concurrent expert transfers do not overwrite inputs.
    static constexpr int kWays = 16;
    struct Miss { int expert, slot, pin_slot; int outgoing_key = -1; };

    Gguf* gg = nullptr;
    uint64_t base[3] = {0, 0, 0};      ///< file offset of the gate / up / down expert tensor
    std::size_t stride[3] = {0, 0, 0}; ///< bytes per expert in each
    void* pin[kWays][3] = {};          ///< the staging; the destination is the caller's arena
    std::size_t pin_bytes[kWays][3] = {};
    cudaStream_t copy{};               ///< so a fetch never queues behind, or in front of, the arithmetic
    cudaEvent_t ev_copy[1] = {};       ///< the previous group's staging has been read out of the pinned
                                       ///< buffers, so this group may overwrite them
    cudaEvent_t ev_arena{};            ///< the compute stream has finished with the previous group's slots
    cudaEvent_t ev_group{};            ///< this group's staging and its index have landed
    Buf idx_dev;                       ///< the device index list the matvecs read
    std::vector<int32_t> idx_host;
    std::vector<Miss> miss;            ///< this group's, reused so the plan costs no allocation
    std::unique_ptr<Pool> pool;
    std::size_t host_stride = 0;       ///< per-position slice of `host`, for the no-mapping fallback
    std::size_t plane_h[3] = {0, 0, 0};
    bool ev_copy_set[1] = {};
    std::vector<uint8_t> host;         ///< the fallback when there is no mapping
    bool ok = false;

    // Cache individual experts in VRAM so different layers can share the capacity.
    Buf tier;
    std::size_t tier_capacity = 0;
    static constexpr int kMaxPools = 4;
    struct TierPool {
        std::size_t stride_r[3] = {0, 0, 0};   ///< bytes between one expert and the next, per role
        std::size_t plane[3] = {0, 0, 0};      ///< absolute byte offset of each role's plane in `tier`
        std::size_t slot_bytes = 0;
        int g0 = 0;          ///< this pool's first slot in the GLOBAL slot id space (key_of/used/claimed)
        int n = 0;           ///< slots in this pool
        size_t ram_spare = 0; // empty pinned host slot for a direct GPU eviction in this pool
        int next_free = 0;   ///< compulsory-miss fill, pool-local
        int hand = 0;        ///< CLOCK hand, pool-local
        int hot_hand = 0;    ///< bounded frequency candidate cursor, pool-local
        int overflow = 0;    ///< rotating slots at the pool's end, for misses the pool cannot keep
        unsigned overflow_at = 0;
    };
    TierPool pools[kMaxPools];
    int n_pools = 0;
    std::vector<int> pool_of;              ///< [n_block] -> pool index, -1 for a block with no experts
    std::vector<int> block_base;           ///< [n_block] -> this block's first POOL-LOCAL slot, -1 if none
    bool pools_full = false;               ///< every pool got a slot for every expert of its blocks
    int pool_i = -1;                       ///< the pool of the block `open_block` last opened
    std::size_t slot_bytes = 0;            ///< the widest pool's, and the "tier is open" sentinel
    int n_slots = 0;                       ///< summed over the pools
    int n_expert = 0;
    int block = -1;                        ///< whose experts `fetch` is being asked for
    std::vector<int32_t> slot_of;          ///< [n_block * n_expert] -> POOL-LOCAL slot, -1 = not resident
    bool exclusive = false;
    std::vector<uint64_t> ram_offset; // keyed by block and expert; offsets into the compact host pool
    std::unique_ptr<uint8_t[]> ram_pool; // pageable fallback
    uint8_t* pinned_ram = nullptr;
    uint8_t* exchange = nullptr; // two pinned expert blobs, reused after the copy stream completes
    uint64_t exchanges = 0;
    cudaStream_t plane_streams[2]{};
    cudaEvent_t planes_ready{};

    uint8_t* ram_plane(int key, int role) const {
        const uint64_t offset = ram_offset[key];
        if (offset == core::detail::kNoCacheComplement) return nullptr;
        const auto& layout = pools[pool_of[key / n_expert]];
        size_t plane = 0;
        for (int r = 0; r < role; ++r) plane += layout.stride_r[r];
        return (pinned_ram ? pinned_ram : ram_pool.get()) + offset + plane;
    }

    bool read_plane(const Gguf& file, int key, int role, uint64_t offset, void* dst, size_t bytes) const {
        if (exclusive) {
            if (const auto* src = ram_plane(key, role)) {
                std::memcpy(dst, src, bytes);
                return true;
            }
            return false;
        }
        if (const auto* src = file.at(offset, bytes)) {
            std::memcpy(dst, src, bytes);
            return true;
        }
        return file.pread_at(offset, dst, bytes);
    }

    // Reuse the group's copy stream and one spare RAM slot per quantization layout.
    // The first victim lands in the spare, then each upload frees its source for
    // the next victim. Stream ordering protects the source even across layers.
    bool transfer_pinned_group(int layer,
                               const size_t sizes[3], const std::vector<Miss>& missing) {
        auto& q = pools[pool_of[layer]];
        size_t spare = q.ram_spare;
        const auto absent = core::detail::kNoCacheComplement;
        // Each expert plane owns disjoint RAM and VRAM bytes. Keep each plane's
        // eviction/upload chain ordered, but let the PCIe copy engines overlap
        // opposite directions on different planes, without adding a host slot.
        const bool overlap = planes_ready && std::any_of(missing.begin(),missing.end(),
            [](const Miss& m){return m.outgoing_key>=0;});
        if (overlap) {
            if (cudaEventRecord(planes_ready,copy)!=cudaSuccess) return false;
            for (auto stream : plane_streams)
                if (cudaStreamWaitEvent(stream,planes_ready,0)!=cudaSuccess) return false;
        }
        for (const Miss& m : missing) {
            const int key = layer * n_expert + m.expert;
            const auto incoming = ram_offset[key];
            if (incoming == absent) return false;
            const bool swap = m.outgoing_key >= 0;
            if (swap && ram_offset[m.outgoing_key] != absent) return false;
            size_t plane = 0;
            for (int r = 0; r < 3; ++r) {
                cudaStream_t dma = overlap && r ? plane_streams[r-1] : copy;
                auto* gpu = static_cast<uint8_t*>(tier.p) + q.plane[r] + size_t(m.slot) * q.stride_r[r];
                const void* src = pinned_ram + incoming + plane;
                if (swap && cudaMemcpyAsync(pinned_ram + spare + plane, gpu, q.stride_r[r],
                                            cudaMemcpyDeviceToHost, dma) != cudaSuccess) return false;
                if (cudaMemcpyAsync(gpu, src, sizes[r], cudaMemcpyHostToDevice, dma) != cudaSuccess)
                    return false;
                plane += q.stride_r[r];
            }
            if (swap) spare = incoming;
        }
        if (cudaStreamSynchronize(copy) != cudaSuccess) return false;
        if (overlap) for (auto stream : plane_streams)
            if (cudaStreamSynchronize(stream)!=cudaSuccess) return false;

        // All workers that consume the owner map see the group only after DMA.
        for (const Miss& m : missing) {
            if (m.outgoing_key < 0) continue; // transient overflow does not change permanent ownership
            auto& incoming = ram_offset[layer * n_expert + m.expert];
            ram_offset[m.outgoing_key] = q.ram_spare;
            q.ram_spare = incoming;
            incoming = absent;
            ++exchanges;
        }
        h2d_bytes += missing.size() * (sizes[0] + sizes[1] + sizes[2]);
        n_fetch += (long) missing.size();
        return true;
    }

    // CPU-selected experts and protected GPU hits are disjoint from this list. The device
    // owner map is published by the caller only after the copy stream has completed.
    bool transfer_exclusive(int layer,
                            const size_t sizes[3], const std::vector<Miss>& missing) {
        if (pinned_ram) return transfer_pinned_group(layer, sizes, missing);
        const auto& q = pools[pool_of[layer]];
        for (const Miss& m : missing) {
            const int key = layer * n_expert + m.expert;
            if (ram_offset[key] == core::detail::kNoCacheComplement) return false;
            const bool swap = m.outgoing_key >= 0;
            size_t pos = 0;
            for (int r = 0; r < 3; ++r) {
                auto* gpu = static_cast<uint8_t*>(tier.p) + q.plane[r] + size_t(m.slot) * q.stride_r[r];
                std::memcpy(exchange + pos, ram_plane(key, r), sizes[r]);
                if (swap && cudaMemcpyAsync(exchange + slot_bytes + pos, gpu, q.stride_r[r],
                                            cudaMemcpyDeviceToHost, copy) != cudaSuccess) return false;
                if (cudaMemcpyAsync(gpu, exchange + pos, sizes[r], cudaMemcpyHostToDevice, copy) != cudaSuccess)
                    return false;
                pos += q.stride_r[r];
            }
            if (cudaStreamSynchronize(copy) != cudaSuccess) return false;
            if (swap) {
                std::memcpy(ram_pool.get() + ram_offset[key], exchange + slot_bytes, q.slot_bytes);
                if (!core::detail::exchange_cache_complement(ram_offset, key, m.outgoing_key)) return false;
                ++exchanges;
            }
            h2d_bytes += sizes[0] + sizes[1] + sizes[2];
            ++n_fetch;
        }
        return true;
    }
    /// Every (block, expert) of every routing block is in a slot, so no token needs a host decision at all
    /// and `latent_moe` can take the router's device ids as its index. Set by `prefill_all`.
    bool complete = false;
    Buf slot_dev;                          ///< device copy of `slot_of`, uploaded once when `complete`
    std::size_t tier_bytes() const {
        std::size_t b = 0;
        for (int p = 0; p < n_pools; ++p) b += (std::size_t) pools[p].n * pools[p].slot_bytes;
        return b;
    }
    /// Slots holding an expert, summed over the pools.
    int filled() const {
        int n = 0;
        for (int p = 0; p < n_pools; ++p) n += pools[p].next_free;
        return n;
    }
    // CLOCK gives recently used entries a second chance before eviction.
    std::vector<int32_t> key_of;           ///< [n_slots], which (block, expert) a slot holds, -1 = free
    std::vector<uint8_t> used;             ///< [n_slots], the second chance
    long evictions = 0;
    bool hot_cache = false, hot_cpu_cold_first = false;
    ExpertHeat heat;
    unsigned long long hot_rejected = 0, hot_admitted = 0;
    std::vector<uint32_t> claimed;         ///< the group that last took a slot, so CLOCK cannot evict one
    uint32_t group_mark = 0;               ///< this group's stamp
    long hits = 0, misses = 0;
    unsigned long long h2d_bytes = 0, d2d_bytes = 0;
    // Where a fetch's own time goes, so "the expert loop is slow" can be attributed instead of guessed at.
    double t_wait = 0.0;   ///< blocked on this way's previous copy
    double t_read = 0.0;   ///< the memcpy out of the mapping, which is where a cold page shows up
    double t_post = 0.0;   ///< enqueueing the three copies and the events
    long n_fetch = 0;


    struct AsyncLayer {
        Gguf* file=nullptr; int block=-1,pool=-1;
        uint64_t base[3]={}; std::size_t stride[3]={};
        char* destination[3]={}; std::size_t destination_stride[3]={};
        std::size_t host_stride=0,host_plane[3]={};
        ExpertAsyncMailbox* host=nullptr; ExpertAsyncGroup group{};
        std::atomic<uint32_t> served{0};
        cpu::NativeFmt cpu_fmt{};
        float* cpu_input_host=nullptr;
        float* cpu_output_host=nullptr;
        float* cpu_output_device=nullptr;
        int cpu_dim=0;
        unsigned long long* split_stats_host=nullptr;   ///< mapped [ta, tw]; null unless a split mode asked
        ~AsyncLayer(){
            // stop_async has joined supply workers and drained all GPU work before descriptors retire.
            if(group.shared_stream)cudaStreamSynchronize(static_cast<cudaStream_t>(group.shared_stream));
            if(group.shared_fork_event)cudaEventDestroy(static_cast<cudaEvent_t>(group.shared_fork_event));
            if(group.shared_join_event)cudaEventDestroy(static_cast<cudaEvent_t>(group.shared_join_event));
            if(group.shared_stream)cudaStreamDestroy(static_cast<cudaStream_t>(group.shared_stream));
            if(group.shared_q8)cudaFree(group.shared_q8);
            if(split_stats_host)cudaFreeHost(split_stats_host);
            if(cpu_output_device)cudaFree(cpu_output_device);
            if(cpu_output_host)cudaFreeHost(cpu_output_host);
            if(cpu_input_host)cudaFreeHost(cpu_input_host);
            if(group.hit_index)cudaFree(group.hit_index);if(group.miss_index)cudaFree(group.miss_index);if(host)cudaFreeHost(host);
        }
    };
    std::vector<std::unique_ptr<AsyncLayer>> async_layers;
    std::thread async_thread;
    struct alignas(64) ReadyPlane { std::atomic<uint32_t> value{0}; };
    ReadyPlane producer_ready[kWays*3];
    std::thread async_producer;
    std::mutex producer_mutex;
    std::condition_variable producer_cv;
    bool producer_stop=false,producer_pending=false,async_parallel_pipeline=false;
    AsyncLayer* producer_layer=nullptr;
    std::vector<Miss> producer_misses;
    const void* producer_sources[kWays][3]={};
    std::atomic<bool> producer_done{true},producer_bad{false};
    long producer_groups=0,producer_experts=0;
    int producer_abort_plane=-1;
    std::atomic<bool> async_stop{false},async_started{false},async_active{false};
    std::atomic<uint32_t> async_failed{0};
    bool async_initialized=false,async_direct=false;
    int async_device=0,async_k=16,async_abort_layer=-1;
    uint32_t async_abort_epoch=1;
    // The CPU pool belongs to its own coordinator, never to the staging Pool.
    // One layer is in flight: GPU done includes the CPU merge before the cursor advances.
    std::unique_ptr<CpuExperts> cpu_experts;
    std::thread cpu_thread;
    std::mutex cpu_mutex;
    std::condition_variable cpu_cv;
    bool cpu_stop=false, cpu_pending=false;
    AsyncLayer* cpu_layer=nullptr;
    uint32_t cpu_epoch=0, cpu_mask=0;
    int cpu_ids[kWays]={};
    int cpu_max=0;
    bool cpu_cache_first=true;
    // Split cache misses between CPU execution and GPU uploads using measured branch costs.
    enum class CpuSplit { Fixed, Ratio, Auto };
    CpuSplit cpu_split=CpuSplit::Fixed;
    double cpu_gpu_frac=0.55;     ///< the GPU's share of the misses for `ratio`, and `auto`'s cold start
    strata::hy4_flash::PcieModel cpu_cost{};   ///< owned by async_loop: it alone picks, samples and solves
    CpuSplitFit cpu_fit{};
    CpuSplitProbe cpu_probe{};
    // The last decision, written by `choose_cpu` and read by `async_loop` when it samples. Both run on the
    // async thread, in that order, within one layer - no synchronisation and no staleness across layers.
    int split_nmiss=0;            ///< misses this layer had
    int split_cpu_n=0;            ///< experts actually given to the CPU, after every clamp
    // Measure CPU work through publication of the resulting rows to device memory.
    std::chrono::steady_clock::time_point cpu_submit_at{};
    std::atomic<unsigned long long> cpu_branch_us{0};   ///< published by the worker, read after `done`
    std::atomic<uint32_t> cpu_branch_epoch{0};          ///< which epoch `cpu_branch_us` belongs to
    unsigned long long split_samples=0,split_solves=0,split_fallbacks=0;
    bool cpu_gpu_quant=true; // match the GPU activation contract; native Q8_K is an audit control
    cudaStream_t cpu_copy=nullptr;
    std::atomic<unsigned long long> cpu_jobs{0}, cpu_result_bytes{0};
    std::atomic<unsigned long long> cpu_compute_us{0};
    bool async_device_plan=false,async_map_dirty=true;
    int32_t* async_device_slots=nullptr;
    std::size_t async_map_count=0;
    uint32_t* async_epoch_arena=nullptr; // one allocation; descriptor epochs never reset between requests
    bool async_device_epoch=false;
    unsigned long long async_fast_groups=0,async_fallback_groups=0,async_false_negative_groups=0;
    unsigned long long async_map_bytes=0,async_map_uploads=0,async_map_reentries=0;

    bool upload_device_map() {
        if(!async_device_plan)return true;
        if(!async_device_slots || async_map_count!=slot_of.size())return false;
        OffloadNvtx mark("hy4-flash.async_map_h2d");
        const std::size_t bytes=async_map_count*sizeof(int32_t);
        if(cudaMemcpyAsync(async_device_slots,slot_of.data(),bytes,cudaMemcpyHostToDevice,copy)!=cudaSuccess ||
           cudaStreamSynchronize(copy)!=cudaSuccess)return false;
        async_map_bytes+=bytes;++async_map_uploads;
        return true;
    }
    void report_device_plan() const {
        std::fprintf(stderr,"HY4_FLASH_DEVICE_PLAN_COUNTERS fast=%llu fallback=%llu false_negative=%llu map_uploads=%llu map_bytes=%llu reentries=%llu\n",
                     async_fast_groups,async_fallback_groups,async_false_negative_groups,async_map_uploads,async_map_bytes,async_map_reentries);
    }


    static uint32_t flag_load(const volatile uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
    static void flag_store(volatile uint32_t* p, uint32_t x) { std::atomic_thread_fence(std::memory_order_seq_cst); __atomic_store_n(p, x, __ATOMIC_RELEASE); }

    // CPU metadata has exactly one owner during decode. Claim every existing reader before CLOCK
    // chooses any write destination; router rank order must not make a later hit evictable.
    bool plan_slots(int block_id, int pi, const int32_t* ids, int k,
                    std::vector<int32_t>& indices, std::vector<uint8_t>& is_hit,
                    std::vector<Miss>& missing, uint32_t on_cpu=0, bool hot_admission=false) {
        if (pi < 0 || pi >= n_pools || k <= 0 || k > kWays) return false;
        TierPool& q = pools[pi]; const int permanent = q.n - q.overflow;
        if (++group_mark == 0) { std::fill(claimed.begin(), claimed.end(), 0); group_mark = 1; }
        indices.assign(k, -1); is_hit.assign(k, 0); missing.clear();
        if (hot_cache) {
            for (int j = 0; j < k; ++j) {
                if (ids[j] < 0 || ids[j] >= n_expert) return false;
                const std::size_t key = (std::size_t)block_id * n_expert + ids[j];
                if (key >= slot_of.size()) return false;
                heat.observe(key); // include CPU-selected misses, before any admission decision
            }
        }
        for (int j = 0; j < k; ++j) {
            if (ids[j] < 0 || ids[j] >= n_expert) return false;
            const std::size_t key = (std::size_t)block_id * n_expert + ids[j];
            if (key >= slot_of.size()) return false;
            const int slot = slot_of[key];
            if (slot < 0) continue;
            if (slot >= permanent || key_of[(std::size_t)(q.g0 + slot)] != (int32_t)key) return false;
            indices[j] = slot; is_hit[j] = 1;
            claimed[(std::size_t)(q.g0 + slot)] = group_mark;
            used[(std::size_t)(q.g0 + slot)] = 1;
        }
        for (int j = 0; j < k; ++j) {
            if (is_hit[j]) { ++hits; continue; }
            ++misses;
            if (on_cpu & (1u << j)) continue; // no victim, slot or resident-map entry for CPU work
            int slot = -1, outgoing_key = -1;
            if (q.next_free < permanent) slot = q.next_free++;
            else {
                if (hot_cache && hot_admission) {
                    const std::size_t key = (std::size_t)block_id * n_expert + ids[j];
                    slot = heat.victim(key, q.g0, permanent, q.hot_hand, key_of, claimed, group_mark);
                    if (slot < 0) ++hot_rejected; else ++hot_admitted;
                } else for (int scan = 0; scan < 2 * permanent; ++scan) {
                    const int cand = q.hand; q.hand = (q.hand + 1) % permanent;
                    if (claimed[(std::size_t)(q.g0+cand)] == group_mark) continue;
                    if (used[(std::size_t)(q.g0+cand)]) { used[(std::size_t)(q.g0+cand)] = 0; continue; }
                    slot = cand; break;
                }
                if (slot >= 0) {
                    const int32_t old = key_of[(std::size_t)(q.g0+slot)];
                    outgoing_key = old;
                    if (old >= 0 && (std::size_t)old < slot_of.size()) { slot_of[(std::size_t)old] = -1; ++evictions; }
                }
            }
            if (slot < 0) {
                if (q.overflow <= 0) return false;
                slot = permanent + (int)(q.overflow_at++ % (unsigned)q.overflow);
            } else {
                const std::size_t key = (std::size_t)block_id * n_expert + ids[j];
                slot_of[key] = slot; key_of[(std::size_t)(q.g0+slot)] = (int32_t)key;
                used[(std::size_t)(q.g0+slot)] = 1; claimed[(std::size_t)(q.g0+slot)] = group_mark;
            }
            indices[j] = slot; missing.push_back({ids[j],slot,(int)missing.size(),outgoing_key});
        }
        for (int j=0;j<k;++j) {
            if (on_cpu & (1u << j)) {
                if (is_hit[j] || indices[j] != -1) return false;
                continue;
            }
            const int slot=indices[j]; bool valid=slot>=0 && slot<q.n;
            if (valid && slot<permanent) valid=key_of[(std::size_t)(q.g0+slot)]==block_id*n_expert+ids[j];
            else if(valid) { valid=false; for(const Miss& m:missing) if(m.slot==slot && m.expert==ids[j]) valid=true; }
            if(!valid) { std::fprintf(stderr,"CAUSAL_WRONG_SLOT block=%d rank=%d slot=%d\n",block_id,j,slot); return false; }
        }
        return true;
    }

    void async_fail(uint32_t code, const char* why) {
        uint32_t zero=0;
        if (async_failed.compare_exchange_strong(zero,code))
            std::fprintf(stderr,"HY4_FLASH_ASYNC_ERROR code=%u %s\n",code,why);
        for (auto& p:async_layers) if(p && p->host) flag_store(&p->host->error,code);
    }
    bool async_error() const { return async_failed.load(std::memory_order_acquire)!=0; }

    void producer_loop() {
        std::vector<Miss> job;job.reserve(kWays);
        for(;;) {
            AsyncLayer* a=nullptr;const void* sources[kWays][3]={};
            {
                std::unique_lock<std::mutex> lock(producer_mutex);
                producer_cv.wait(lock,[this]{return producer_stop||producer_pending;});
                if(producer_stop)return;
                a=producer_layer;job=producer_misses;
                std::memcpy(sources,producer_sources,sizeof(sources));
                producer_pending=false;
            }
            {
                OffloadNvtx producer_mark("hy4-flash.async_parallel_producer");
                pool->run((int)job.size(),[&](int i){
                    const Miss& m=job[(std::size_t)i];
                    for(int r=0;r<3;++r) {
                        if(producer_bad.load(std::memory_order_acquire))return;
                        const void* src=sources[m.pin_slot][r];
                        if(!src) {
                            auto* tmp=host.data()+(std::size_t)m.pin_slot*a->host_stride+a->host_plane[r];
                            if(!a->file->pread_at(a->base[r]+(uint64_t)m.expert*a->stride[r],tmp,a->stride[r])){
                                producer_bad.store(true,std::memory_order_release);return;
                            }
                            src=tmp;
                        }
                        {
                            OffloadNvtx mark("hy4-flash.async_parallel_plane_memcpy");
                            std::memcpy(pin[m.pin_slot][r],src,a->stride[r]);
                        }
                        producer_ready[m.pin_slot*3+r].value.store(1,std::memory_order_release);
                        if(a->block==1 && m.pin_slot*3+r==producer_abort_plane){
                            std::fprintf(stderr,"HY4_FLASH_ASYNC_PRODUCER_ABORT block=%d plane=%d\n",a->block,producer_abort_plane);
                            producer_bad.store(true,std::memory_order_release);return;
                        }
                    }
                });
            }
            // All pool callbacks have returned, not merely published their last plane. No callback
            // retains references to the immutable command when service is allowed to reuse it.
            producer_done.store(true,std::memory_order_release);
        }
    }
    bool stage_parallel_planes(AsyncLayer& a,const std::vector<Miss>& missing,const void* sources[kWays][3]) {
        if(!async_producer.joinable() || !producer_done.load(std::memory_order_acquire))return false;
        for(auto& ready:producer_ready)ready.value.store(0,std::memory_order_relaxed);
        producer_bad.store(false,std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(producer_mutex);
            producer_layer=&a;producer_misses=missing;
            std::memcpy(producer_sources,sources,sizeof(producer_sources));
            producer_done.store(false,std::memory_order_release);
            producer_pending=true;
        }
        producer_cv.notify_one();
        bool submitted[kWays*3]={};int left=(int)missing.size()*3;bool failed=false;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        {
            OffloadNvtx supply_mark("hy4-flash.async_parallel_supply");
            while(left && !failed) {
                if(producer_bad.load(std::memory_order_acquire)){failed=true;break;}
                bool progress=false;
                for(const Miss& m:missing)for(int r=0;r<3;++r) {
                    const int index=m.pin_slot*3+r;
                    if(failed || submitted[index] || !producer_ready[index].value.load(std::memory_order_acquire))continue;
                    OffloadNvtx mark("hy4-flash.async_parallel_plane_h2d");
                    if(cudaMemcpyAsync(a.destination[r]+(std::size_t)m.slot*a.destination_stride[r],
                                       pin[m.pin_slot][r],a.stride[r],cudaMemcpyHostToDevice,copy)!=cudaSuccess){failed=true;break;}
                    submitted[index]=true;--left;progress=true;h2d_bytes+=a.stride[r];
                }
                if(std::chrono::steady_clock::now()>deadline){producer_bad.store(true,std::memory_order_release);failed=true;}
                if(!progress)std::this_thread::yield();
            }
        }
        // Even an enqueue error cannot retire or destroy the command while the producer/pool reads it.
        while(!producer_done.load(std::memory_order_acquire))std::this_thread::yield();
        if(failed || producer_bad.load(std::memory_order_acquire))return false;
        ++producer_groups;producer_experts+=(long)missing.size();n_fetch+=(long)missing.size();
        OffloadNvtx finish_mark("hy4-flash.async_parallel_copy_finish");
        return cudaStreamSynchronize(copy)==cudaSuccess;
    }

    bool stage_async(AsyncLayer& a, const std::vector<Miss>& missing) {
        if (missing.empty()) return true;
        if (exclusive) return transfer_exclusive(a.block, a.stride, missing);
        const void* sources[kWays][3] = {};
        bool direct = async_direct;
        for(const Miss& m:missing) for(int r=0;r<3;++r) {
            sources[m.pin_slot][r]=a.file->at(a.base[r]+(uint64_t)m.expert*a.stride[r],a.stride[r]);
            if(!sources[m.pin_slot][r]) direct=false;
        }
        if(!direct && async_parallel_pipeline)return stage_parallel_planes(a,missing,sources);
        if (!direct) {
            OffloadNvtx mark("hy4-flash.async_mmap_to_pinned");
            std::atomic<bool> bad{false};
            pool->run((int)missing.size(),[&](int i){
                const Miss& m=missing[(std::size_t)i];
                for(int r=0;r<3;++r) {
                    const void* src=sources[m.pin_slot][r];
                    if(!src) {
                        auto* tmp=host.data()+(std::size_t)m.pin_slot*a.host_stride+a.host_plane[r];
                        if(!a.file->pread_at(a.base[r]+(uint64_t)m.expert*a.stride[r],tmp,a.stride[r])) {bad.store(true);return;}
                        src=tmp;
                    }
                    std::memcpy(pin[m.pin_slot][r],src,a.stride[r]);
                }
            });
            if(bad.load()) return false;
        }
        // Independent stream only. Recording an event on a compute graph waiting for B would deadlock.
        OffloadNvtx mark("hy4-flash.async_h2d");
        for(const Miss& m:missing) for(int r=0;r<3;++r) {
            const void* src=direct?sources[m.pin_slot][r]:pin[m.pin_slot][r];
            if(a.file->copy_expert_async(a.destination[r]+(std::size_t)m.slot*a.destination_stride[r],
                                         src,a.stride[r],copy)!=cudaSuccess) return false;
            h2d_bytes+=a.stride[r];
        }
        n_fetch+=(long)missing.size();
        // B is completion, not submission. This CPU worker may block while GPU resident work runs.
        return cudaStreamSynchronize(copy)==cudaSuccess;
    }

    uint32_t choose_cpu(const AsyncLayer& a, const int32_t* ids) {
        if (!cpu_max) return 0;
        int limit=cpu_max;
        // The missed experts of this layer. `fixed` only needs this under cache-first, but `ratio` and
        // `auto` size themselves from it, so it is counted once here and reused. A decode step's top-16
        // holds no repeats (the router picks distinct experts), so this count IS the distinct count.
        int missing=0;
        for(int j=0;j<async_k;++j) {
            const int id=ids[j];
            if(id>=0 && id<n_expert && slot_of[(std::size_t)a.block*n_expert+id]<0)++missing;
        }
        // ---- THE COUNT, from the mode. `fixed` leaves `limit` at the cap, exactly as before.
        if(cpu_split!=CpuSplit::Fixed && missing>0) {
            // `ratio`'s count, which is also what `auto` falls back to before the fit has samples.
            const int gpu_ratio=(int)((double)missing*cpu_gpu_frac);
            int gpu=gpu_ratio;
            if(cpu_split==CpuSplit::Auto) {
                // `pick` returns the GPU's count and reads the fit that `async_loop` solved; it is the one
                // thread that touches `cpu_cost`, and `choose_cpu` runs on it.
                gpu=cpu_cost.pick(missing,gpu_ratio);
                if(gpu<0)gpu=0; else if(gpu>missing)gpu=missing;
            }
            limit=std::min(limit,missing-gpu);
            if(limit<0)limit=0;
        }
        int cpu_capacity=std::min(cpu_max,missing);
        if(cpu_cache_first) {
            const TierPool& q=pools[a.pool];
            const int free=std::max(0,q.n-q.overflow-q.next_free);
            // Fill unused permanent slots first so CPU offload does not prevent cache warmup.
            cpu_capacity=std::min(cpu_capacity,std::max(0,missing-free));
            limit=std::min(limit,cpu_capacity);
        }
        if(cpu_probe.enabled) limit=cpu_probe.choose(limit,cpu_capacity,cpu_cost);
        split_nmiss=missing;
        uint32_t mask=0; int n=0;
        if (hot_cache && hot_cpu_cold_first) {
            // Independent opt-in: give the CPU the coldest observed misses,
            // leaving popular misses eligible for GPU cache admission. Ties
            // preserve the old last-rank-first choice. Free slots still win.
            for (; n < limit; ++n) {
                int best = -1; uint16_t least = 0;
                for (int j = async_k - 1; j >= 0; --j) {
                    if (mask & (1u << j)) continue;
                    const int id = ids[j];
                    if (id < 0 || id >= n_expert) continue;
                    const std::size_t key = (std::size_t)a.block * n_expert + id;
                    if (slot_of[key] >= 0) continue;
                    const uint16_t score = heat.score(key);
                    if (best < 0 || score < least) { best = j; least = score; }
                }
                if (best < 0) break;
                mask |= 1u << best;
            }
            split_cpu_n=n;
            return mask;
        }
        // Retain GPU hits. Route the last missing ranks to the CPU up to the
        // explicit cap; every selected expert is still evaluated exactly once.
        for(int j=async_k-1;j>=0 && n<limit;--j) {
            const int id=ids[j];
            if(id<0 || id>=n_expert) continue; // plan_slots reports invalid routing
            if(slot_of[(std::size_t)a.block*n_expert+id]<0){mask|=1u<<j;++n;}
        }
        split_cpu_n=n;
        return mask;
    }
    bool submit_cpu(AsyncLayer& a,uint32_t epoch,uint32_t mask,const int32_t* ids) {
        std::lock_guard<std::mutex> lock(cpu_mutex);
        if(cpu_pending || cpu_stop)return false;
        cpu_layer=&a;cpu_epoch=epoch;cpu_mask=mask;
        std::copy(ids,ids+async_k,cpu_ids);
        // The branch's clock starts where the plan becomes the worker's, under the same lock that hands it
        // over, so the worker cannot read a timestamp from the layer before.
        cpu_submit_at=std::chrono::steady_clock::now();
        cpu_pending=true;cpu_cv.notify_one();return true;
    }
    void cpu_loop() {
        if(cudaSetDevice(async_device)!=cudaSuccess){async_fail(20,"CPU worker cudaSetDevice");return;}
        for(;;) {
            AsyncLayer* a;uint32_t epoch,mask;int ids[kWays];
            std::chrono::steady_clock::time_point submitted{};
            {
                std::unique_lock<std::mutex> lock(cpu_mutex);
                cpu_cv.wait(lock,[&]{return cpu_stop || cpu_pending;});
                if(cpu_stop)return;
                a=cpu_layer;epoch=cpu_epoch;mask=cpu_mask;
                std::copy(cpu_ids,cpu_ids+async_k,ids);cpu_pending=false;
                submitted=cpu_submit_at;
            }
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(flag_load(&a->host->input_ready)<epoch && !async_error() && !async_stop.load()) {
                if(std::chrono::steady_clock::now()>deadline){async_fail(21,"CPU input timeout");break;}
                std::this_thread::yield();
            }
            if(async_error() || async_stop.load())continue;
            try {
                const uint8_t* gates[kWays]={};const uint8_t* ups[kWays]={};const uint8_t* downs[kWays]={};
                int ranks[kWays],n=0;
                for(int j=0;j<async_k;++j)if(mask&(1u<<j)) {
                    const int job=n++;
                    ranks[job]=j;
                    if (exclusive) {
                        const int key = a->block * n_expert + ids[j];
                        gates[job] = ram_plane(key, 0);
                        ups[job] = ram_plane(key, 1);
                        downs[job] = ram_plane(key, 2);
                    } else {
                        gates[job]=static_cast<const uint8_t*>(a->file->at(a->base[0]+(uint64_t)ids[j]*a->stride[0],a->stride[0]));
                        ups[job]=static_cast<const uint8_t*>(a->file->at(a->base[1]+(uint64_t)ids[j]*a->stride[1],a->stride[1]));
                        downs[job]=static_cast<const uint8_t*>(a->file->at(a->base[2]+(uint64_t)ids[j]*a->stride[2],a->stride[2]));
                    }
                    if(!gates[job] || !ups[job] || !downs[job])throw std::runtime_error("CPU expert weights are unavailable");
                }
                std::string err;
                const auto begin=std::chrono::steady_clock::now();
                if(!cpu_experts->compute(a->cpu_fmt,gates,ups,downs,ranks,n,a->cpu_input_host,a->cpu_output_host,err))
                    throw std::runtime_error(err);
                cpu_compute_us.fetch_add((unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now()-begin).count(),std::memory_order_relaxed);
                // Opt-in arithmetic audit of the first routed layer/epoch. Disabled
                // during timing; these are unweighted CPU rows for the same F32 input.
                static const char* diag_dir=std::getenv("HY4_FLASH_CPU_DIAG_DIR");
                if(diag_dir && a->block==1 && epoch==1) {
                    const auto dump=[&](const char* name,const void* data,std::size_t bytes) {
                        const std::string path=std::string(diag_dir)+"/"+name;
                        std::FILE* f=std::fopen(path.c_str(),"wb");
                        if(!f)throw std::runtime_error("CPU diagnostic open failed");
                        const bool ok=std::fwrite(data,1,bytes,f)==bytes;
                        std::fclose(f);if(!ok)throw std::runtime_error("CPU diagnostic write failed");
                    };
                    dump("input.f32",a->cpu_input_host,a->cpu_dim*sizeof(float));
                    dump("parts.f32",a->cpu_output_host,kWays*a->cpu_dim*sizeof(float));
                    dump("ids.i32",ids,kWays*sizeof(int));
                    dump("ranks.i32",ranks,n*sizeof(int));
                    const std::string path=std::string(diag_dir)+"/meta.txt";
                    std::FILE* f=std::fopen(path.c_str(),"w");
                    if(!f)throw std::runtime_error("CPU diagnostic metadata open failed");
                    std::fprintf(f,"block %d epoch %u dim %d ff %lld gu_type %d d_type %d clamp %.9g count %d\n",
                        a->block,epoch,a->cpu_dim,(long long)a->cpu_fmt.n_ff,a->cpu_fmt.gu_type,
                        a->cpu_fmt.d_type,(double)a->cpu_fmt.swiglu_clamp,n);
                    for(int r=0;r<3;++r)std::fprintf(f,"plane %d base %llu stride %zu\n",r,(unsigned long long)a->base[r],a->stride[r]);
                    std::fclose(f);
                }
                // Separate copy stream: neither the compute stream waiting on this
                // result nor the expert-weight DMA stream may be synchronized here.
                for(int j=0;j<async_k;++j)if(mask&(1u<<j)) {
                    const std::size_t off=(std::size_t)j*a->cpu_dim;
                    if(cudaMemcpyAsync(a->cpu_output_device+off,a->cpu_output_host+off,
                        a->cpu_dim*sizeof(float),cudaMemcpyHostToDevice,cpu_copy)!=cudaSuccess)
                        throw std::runtime_error("CPU result upload failed");
                }
                if(cudaStreamSynchronize(cpu_copy)!=cudaSuccess)throw std::runtime_error("CPU result DMA failed");
                cpu_jobs.fetch_add(n,std::memory_order_relaxed);
                cpu_result_bytes.fetch_add((unsigned long long)n*a->cpu_dim*sizeof(float),std::memory_order_relaxed);
                // Publish the CPU duration before the epoch that makes the sample visible.
                cpu_branch_us.store((unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now()-submitted).count(),std::memory_order_relaxed);
                cpu_branch_epoch.store(epoch,std::memory_order_release);
                if(!async_error())flag_store(&a->host->cpu_ready,epoch);
            } catch(const std::exception& e) {async_fail(22,e.what());}
            catch(...) {async_fail(22,"CPU expert worker exception");}
        }
    }

    void async_loop() {
        if(cudaSetDevice(async_device)!=cudaSuccess) {async_fail(1,"worker cudaSetDevice");async_started.store(true);return;}
        async_started.store(true,std::memory_order_release);
        std::vector<int32_t> indices;std::vector<uint8_t> hit;std::vector<Miss> missing;
        indices.reserve(kWays);hit.reserve(kWays);missing.reserve(kWays);
        std::size_t cursor=0;
        while(cursor<async_layers.size() && !async_layers[cursor])++cursor;
        if(cursor==async_layers.size()){async_fail(10,"no async layers");return;}
        const std::size_t first=cursor;
        while(!async_stop.load(std::memory_order_acquire)) {
            if(async_error()){async_active.store(false,std::memory_order_release);std::this_thread::yield();continue;}
            // Never skip an unpublished earlier layer: fast GPU layers may already have advanced.
            AsyncLayer& a=*async_layers[cursor];
            const uint32_t request=flag_load(&a.host->request);
            if(request==a.served.load(std::memory_order_acquire)){std::this_thread::yield();continue;}
            async_active.store(true,std::memory_order_release);
            if(request!=a.served.load()+1 || request==0){async_fail(2,"non-monotonic request epoch");break;}
            int32_t ids[kWays],snapshot[kWays];
            const bool fast=async_device_plan && flag_load(&a.host->fast_epoch)==request;
            const volatile int32_t* published_ids=a.host->ids;
            const volatile int32_t* published_hits=a.host->hit;
            for(int j=0;j<async_k;++j){ids[j]=published_ids[j];snapshot[j]=fast?published_hits[j]:-1;}
            if(fast) {
                const TierPool& q=pools[a.pool];const int permanent=q.n-q.overflow;
                bool valid=true;
                for(int j=0;j<async_k;++j) {
                    const int id=ids[j],slot=snapshot[j];
                    if(id<0 || id>=n_expert || slot<0 || slot>=permanent){valid=false;break;}
                    const std::size_t key=(std::size_t)a.block*n_expert+id;
                    if(key>=slot_of.size() || slot_of[key]!=slot || key_of[(std::size_t)(q.g0+slot)]!=(int32_t)key){valid=false;break;}
                }
                if(!valid){async_fail(11,"GPU all-hit owner snapshot mismatch");break;}
            }
            const uint32_t on_cpu=fast?0:choose_cpu(a,ids);
            { OffloadNvtx mark("hy4-flash.async_cache_plan");
              if(!plan_slots(a.block,a.pool,ids,async_k,indices,hit,missing,on_cpu,true)){async_fail(3,"slot plan invariant");break;} }
            if(fast) {
                bool valid=missing.empty();
                for(int j=0;j<async_k;++j)valid=valid && hit[j] && indices[j]==snapshot[j];
                if(!valid){async_fail(12,"GPU all-hit differs from CPU policy");break;}
                ++async_fast_groups;
                // GPU already consumed its indices. Only CPU CLOCK/counters changed above.
            } else {
                ++async_fallback_groups;
                if(async_device_plan && missing.empty() && !on_cpu)++async_false_negative_groups;
                for(int j=0;j<async_k;++j){a.host->hit[j]=hit[j]?indices[j]:-1;a.host->miss[j]=hit[j]?-1:indices[j];}
            }
            a.host->cpu_mask=on_cpu;
            flag_store(&a.host->plan_ready,request);
            if(on_cpu && !submit_cpu(a,request,on_cpu,ids)){async_fail(23,"CPU queue busy");break;}
            const bool inject=async_abort_layer==a.block && request==async_abort_epoch;
            if(inject || (!fast && !stage_async(a,missing))){async_fail(4,inject?"injected supply abort":"staging or DMA failed");break;}
            // The GPU is blocked on this real miss. Publish all cross-layer evictions only after
            // expert DMA completes; later layer probes cannot execute until B and this layer's done.
            if(async_device_plan && !missing.empty() && !upload_device_map()){
                async_fail(13,"device residency map upload failed");break;
            }
            flag_store(&a.host->miss_ready,request);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(flag_load(&a.host->done)!=request && !async_stop.load() && !async_error()) {
                if(std::chrono::steady_clock::now()>deadline){async_fail(7,"GPU done timeout");break;}
                std::this_thread::yield();
            }
            if(async_error())break;
            // Read both timing stamps after the completion fence makes them visible on the host.
            if(cpu_split!=CpuSplit::Fixed && a.split_stats_host!=nullptr && !fast && split_nmiss>0) {
                const unsigned long long ta=a.split_stats_host[0],tw=a.split_stats_host[1];
                const int gpu_n=split_nmiss-split_cpu_n;
                double cpu_ms=0.0;
                bool have_cpu=split_cpu_n==0;
                if(split_cpu_n>0 && cpu_branch_epoch.load(std::memory_order_acquire)==request) {
                    cpu_ms=(double)cpu_branch_us.load(std::memory_order_relaxed)*1e-3;
                    have_cpu=true;
                }
                const bool gpu_ok=ta!=0 && tw>=ta && (tw-ta)<=1000000000ull;
                if(gpu_ok && have_cpu) {
                    if(cpu_probe.enabled) cpu_probe.observe(cpu_cost,split_cpu_n,gpu_n,cpu_ms);
                    cpu_fit.sample(cpu_cost,split_cpu_n,gpu_n,cpu_ms,(double)(tw-ta)*1e-6);
                    ++split_samples;
                    // Update the cost fit after this layer finishes and before assigning its next batch.
                    if((split_samples&15u)==0){cpu_fit.solve(cpu_cost);++split_solves;}
                } else ++split_fallbacks;
                a.split_stats_host[0]=a.split_stats_host[1]=0;   // never pair a stale stamp with a new epoch
            }
            a.served.store(request,std::memory_order_release);
            do {++cursor;} while(cursor<async_layers.size() && !async_layers[cursor]);
            if(cursor==async_layers.size())cursor=first;
            async_active.store(false,std::memory_order_release);
        }
        async_active.store(false,std::memory_order_release);
    }

    bool configure_cpu_experts() {
        if (cpu_experts) return true;
        const char* cap = std::getenv("STRATA_HY4_FLASH_CPU_MISS_MAX");
        cpu_max = cap ? std::atoi(cap) : 0;
        if (cpu_max < 0 || cpu_max > kWays) return false;
        if (!cpu_max) return true;

        const char* cache_first = std::getenv("STRATA_HY4_FLASH_CPU_CACHE_FIRST");
        cpu_cache_first = !cache_first || std::atoi(cache_first) != 0;
        const char* native = std::getenv("STRATA_HY4_FLASH_CPU_NATIVE_Q8K");
        cpu_gpu_quant = !native || std::atoi(native) == 0;
        if (const char* split = std::getenv("STRATA_HY4_FLASH_CPU_SPLIT")) {
            const std::string policy(split);
            if (policy == "fixed") cpu_split = CpuSplit::Fixed;
            else if (policy == "ratio") cpu_split = CpuSplit::Ratio;
            else if (policy == "auto") cpu_split = CpuSplit::Auto;
            else return false;
        }
        if (const char* fraction = std::getenv("STRATA_HY4_FLASH_GPU_MISS_FRAC")) {
            cpu_gpu_frac = std::atof(fraction);
            if (!(cpu_gpu_frac >= 0.0 && cpu_gpu_frac <= 1.0)) return false;
        }
        if (cpu_split == CpuSplit::Auto) {
            if (const char* probe = std::getenv("STRATA_HY4_FLASH_CPU_EXPLORE")) {
                if (std::strcmp(probe,"adaptive")==0) cpu_probe.enabled=true;
                else if (std::strcmp(probe,"default")!=0) return false;
            }
            if (!cpu_fit.configure_auto(std::getenv("STRATA_HY4_FLASH_CPU_FIT"))) return false;
        }
        const char* cw = std::getenv("STRATA_HY4_FLASH_CPU_WORKERS");
        const int workers = cw ? std::atoi(cw) : 8;
        if (workers <= 0) return false;
        cpu_experts = std::make_unique<CpuExperts>(workers);
        cpu_cost.on = cpu_split == CpuSplit::Auto;
        std::fprintf(stderr,"HY4_FLASH_CPU_EXPERTS max_miss=%d workers=%d activation=%s cache_first=%d split=%s\n",
                     cpu_max,workers,cpu_gpu_quant?"Q8_1-compatible":"native-Q8_K",(int)cpu_cache_first,
                     cpu_split==CpuSplit::Fixed?"fixed":cpu_split==CpuSplit::Ratio?"ratio":"auto");
        if (cpu_split != CpuSplit::Fixed)
            std::fprintf(stderr,"HY4_FLASH_CPU_SPLIT mode=%s gpu_frac=%.2f cpu_range=0..%d\n",
                         cpu_split==CpuSplit::Ratio?"ratio":"auto",cpu_gpu_frac,cpu_max);
        return true;
    }
    bool prepare_async(Gguf& file,const std::string&,int il,int hidden_dim,int,int k,float clamp,ExpertAsyncGroup* out,cudaStream_t) {
        if(complete || k!=kWays || il<0 || (std::size_t)il>=pool_of.size())return false;
        if(!async_initialized) {
            if(!ensure_copy() || cudaStreamSynchronize(copy)!=cudaSuccess)return false;
            cudaStreamDestroy(copy); copy=nullptr;
            if(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking)!=cudaSuccess)return false;
            async_k=k;
            if (!configure_cpu_experts()) return false;
            if (cpu_max && cudaStreamCreateWithFlags(&cpu_copy,cudaStreamNonBlocking)!=cudaSuccess)
                return false;
            const char* dp=std::getenv("STRATA_HY4_FLASH_ASYNC_DEVICE_PLAN");async_device_plan=dp&&std::atoi(dp)!=0;
            if(async_device_plan) {
                async_map_count=slot_of.size();
                if(!async_map_count || cudaMalloc((void**)&async_device_slots,async_map_count*sizeof(int32_t))!=cudaSuccess)return false;
            }
            const char* de=std::getenv("STRATA_HY4_FLASH_DEVICE_EPOCH");
            const bool epoch_requested=de&&std::atoi(de)!=0;
            const char* rp=std::getenv("STRATA_HY4_FLASH_ASYNC_ROUTER_PUBLISH");
            async_device_epoch=epoch_requested && async_device_plan && rp && std::atoi(rp)!=0 &&
                               kernels::native_router_sigmoid_supported(n_expert,k);
            for(int pi=0;pi<n_pools && async_device_epoch;++pi)
                if(pools[pi].n<=pools[pi].overflow)async_device_epoch=false;
            // Report whether all prerequisites for the timing epoch were satisfied.
            if(epoch_requested)
                std::fprintf(stderr,"HY4_FLASH_DEVICE_EPOCH requested=1 resolved=%d\n",async_device_epoch?1:0);
            if(async_device_epoch) {
                const std::size_t bytes=pool_of.size()*2*sizeof(uint32_t);
                if(!bytes || cudaMalloc((void**)&async_epoch_arena,bytes)!=cudaSuccess ||
                   cudaMemset(async_epoch_arena,0,bytes)!=cudaSuccess ||
                   cudaDeviceSynchronize()!=cudaSuccess)return false;
                std::fprintf(stderr,"HY4_FLASH_DEVICE_EPOCH active=1 layers=%zu bytes=%zu\n",pool_of.size(),bytes);
            } else if(epoch_requested) {
                std::fprintf(stderr,"HY4_FLASH_DEVICE_EPOCH active=0 fallback=legacy requires-device-plan-fused-router-permanent-slots\n");
            }
            const char* abort_epoch=std::getenv("HY4_FLASH_ASYNC_ABORT_EPOCH");
            async_abort_epoch=abort_epoch?(uint32_t)std::strtoul(abort_epoch,nullptr,10):1;
            if(!async_abort_epoch)return false;

            const char* direct=std::getenv("HY4_FLASH_CAUSAL_DIRECT_MMAP");async_direct=direct&&std::atoi(direct)!=0;
            const char* abort=std::getenv("HY4_FLASH_ASYNC_ABORT_LAYER");async_abort_layer=abort?std::atoi(abort):-1;
            if(cudaGetDevice(&async_device)!=cudaSuccess)return false;
            std::size_t max_stride[3]={};
            for(int pi=0;pi<n_pools;++pi)for(int r=0;r<3;++r)max_stride[r]=std::max(max_stride[r],pools[pi].stride_r[r]);
            for(int w=0;w<kWays;++w)for(int r=0;r<3;++r)if(pin_bytes[w][r]<max_stride[r]) {
                if(pin[w][r])cudaFreeHost(pin[w][r]);pin[w][r]=nullptr;
                if(cudaHostAlloc(&pin[w][r],max_stride[r],cudaHostAllocDefault)!=cudaSuccess)return false;
                pin_bytes[w][r]=max_stride[r];
            }
            const std::size_t max_host=(max_stride[0]+max_stride[1]+max_stride[2])*kWays;
            if(host.size()<max_host)host.resize(max_host);
            async_layers.resize(pool_of.size());
            size_t shared_layers=0,shared_bytes=0;
            // Allocate ALL descriptors/index storage before any graph can wait for a service response.
            for(std::size_t block_id=0;block_id<pool_of.size();++block_id) {
                const int pi=pool_of[block_id];if(pi<0)continue;
                auto a=std::make_unique<AsyncLayer>();a->file=&file;a->block=(int)block_id;a->pool=pi;
                const char* roles[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
                for(int r=0;r<3;++r) {
                    const TensorInfo* ti=file.info("blk."+std::to_string(block_id)+"."+roles[r]);
                    if(!ti || ti->shape.size()<3 || ti->shape[2]!=(uint64_t)n_expert)return false;
                    a->base[r]=file.data_start()+ti->offset;a->stride[r]=(std::size_t)tensor_payload_bytes(*ti)/n_expert;
                    a->destination[r]=static_cast<char*>(tier.p)+pools[pi].plane[r];a->destination_stride[r]=pools[pi].stride_r[r];
                    a->host_plane[r]=a->host_stride;a->host_stride+=a->stride[r];
                }
                if(cudaHostAlloc((void**)&a->host,sizeof(ExpertAsyncMailbox),cudaHostAllocMapped)!=cudaSuccess)return false;
                std::memset(a->host,0,sizeof(ExpertAsyncMailbox));
                // The GPU branch's two stamps, mapped like the mailbox so the device writes them where the
                // host reads them. Allocated per layer and kept for the engine's life, so the address the
                // graph captured stays valid; `fixed` allocates nothing and launches no stamp node.
                if(cpu_max && cpu_split!=CpuSplit::Fixed) {
                    if(cudaHostAlloc((void**)&a->split_stats_host,2*sizeof(unsigned long long),cudaHostAllocMapped)!=cudaSuccess)return false;
                    a->split_stats_host[0]=a->split_stats_host[1]=0;
                    if(cudaHostGetDevicePointer((void**)&a->group.split_stats,a->split_stats_host,0)!=cudaSuccess)return false;
                }
                if(cudaHostGetDevicePointer((void**)&a->group.mailbox,a->host,0)!=cudaSuccess ||
                   cudaMalloc((void**)&a->group.hit_index,k*sizeof(int32_t))!=cudaSuccess ||
                   cudaMalloc((void**)&a->group.miss_index,k*sizeof(int32_t))!=cudaSuccess)return false;
                cudaMemset(a->group.hit_index,0xff,k*sizeof(int32_t));cudaMemset(a->group.miss_index,0xff,k*sizeof(int32_t));
                a->group.planes={a->destination[0],a->destination_stride[0],a->destination[1],a->destination_stride[1],a->destination[2],a->destination_stride[2],nullptr};
                const char* fork_env=std::getenv("STRATA_HY4_FLASH_ASYNC_SHARED_STREAM");
                if(fork_env && std::atoi(fork_env)!=0) {
                    cudaStream_t fork_stream=nullptr;cudaEvent_t fork_event=nullptr,join_event=nullptr;
                    if(cudaStreamCreateWithFlags(&fork_stream,cudaStreamNonBlocking)!=cudaSuccess)return false;
                    a->group.shared_stream=static_cast<void*>(fork_stream);
                    if(cudaEventCreateWithFlags(&fork_event,cudaEventDisableTiming)!=cudaSuccess)return false;
                    a->group.shared_fork_event=static_cast<void*>(fork_event);
                    if(cudaEventCreateWithFlags(&join_event,cudaEventDisableTiming)!=cudaSuccess)return false;
                    a->group.shared_join_event=static_cast<void*>(join_event);
                    const TensorInfo* shared_gate=file.info("blk."+std::to_string(block_id)+".ffn_gate_shexp.weight");
                    const TensorInfo* routed_gate=file.info("blk."+std::to_string(block_id)+".ffn_gate_exps.weight");
                    if(!shared_gate || shared_gate->shape.size()<2 || !routed_gate || routed_gate->shape.size()<2)return false;
                    const int width=std::max(hidden_dim,std::max((int)shared_gate->shape[1],(int)routed_gate->shape[1]));
                    const std::size_t qbytes=kernels::native_q8_1_bytes(width,1);
                    if(cudaMalloc(&a->group.shared_q8,qbytes)!=cudaSuccess)return false;
                    ++shared_layers;shared_bytes+=qbytes;
                }

                if(cpu_max) {
                    const TensorInfo* gate=file.info("blk."+std::to_string(block_id)+".ffn_gate_exps.weight");
                    const TensorInfo* up=file.info("blk."+std::to_string(block_id)+".ffn_up_exps.weight");
                    const TensorInfo* down=file.info("blk."+std::to_string(block_id)+".ffn_down_exps.weight");
                    std::string why;
                    if(!gate || !up || !down || gate->type!=up->type ||
                        !CpuExperts::make_format(gate->type,down->type,gate->shape[0],gate->shape[1],clamp,a->cpu_fmt,why,cpu_gpu_quant)) {
                        std::fprintf(stderr,"HY4_FLASH_CPU_ERROR unsupported format: %s\n",why.c_str());return false;
                    }
                    a->cpu_dim=(int)gate->shape[0];
                    const std::size_t input_bytes=(std::size_t)a->cpu_dim*sizeof(float);
                    const std::size_t result_bytes=input_bytes*kWays;
                    if(cudaHostAlloc((void**)&a->cpu_input_host,input_bytes,cudaHostAllocMapped)!=cudaSuccess ||
                        cudaHostGetDevicePointer((void**)&a->group.cpu_input,a->cpu_input_host,0)!=cudaSuccess ||
                        cudaHostAlloc((void**)&a->cpu_output_host,result_bytes,cudaHostAllocDefault)!=cudaSuccess ||
                        cudaMalloc((void**)&a->cpu_output_device,result_bytes)!=cudaSuccess)return false;
                    std::memset(a->cpu_output_host,0,result_bytes);
                    a->group.cpu_parts=a->cpu_output_device;
                }

                if(async_device_plan) {
                    const std::size_t start=block_id*(std::size_t)n_expert;
                    if(start+(std::size_t)n_expert>async_map_count)return false;
                    a->group.device_slot_of=async_device_slots+start;
                    a->group.n_expert=n_expert;
                    a->group.permanent_slots=pools[pi].n-pools[pi].overflow;
                }
                if(async_device_epoch)a->group.device_epoch=async_epoch_arena+block_id*2;
                async_layers[block_id]=std::move(a);
            }
            const char* pin_experts = std::getenv("STRATA_HY4_FLASH_PIN_EXPERT_MMAP");
            if (!exclusive && pin_experts && std::atoi(pin_experts) != 0) {
                // The first forward registered the source during initialization, before prefill.
                // Never defer multi-GiB host registration into a decode step or graph capture.
                async_direct = file.expert_pin_ready();
                std::fprintf(stderr, "HY4_FLASH_PIN_EXPERT_MMAP supply=%s\n",
                    async_direct ? "registered-direct-three-plane" : "staged-fallback");
            }
            if(shared_layers)std::fprintf(stderr,"HY4_FLASH_ASYNC_SHARED_RESOURCES layers=%zu q8_bytes=%zu\n",shared_layers,shared_bytes);
            const char* pipeline_env=std::getenv("HY4_FLASH_ASYNC_PARALLEL_PIPELINE");
            async_parallel_pipeline=pipeline_env&&std::atoi(pipeline_env)!=0;
            const char* producer_abort_env=std::getenv("HY4_FLASH_ASYNC_PRODUCER_ABORT_PLANE");producer_abort_plane=producer_abort_env?std::atoi(producer_abort_env):-1;
            if(async_parallel_pipeline){producer_misses.reserve(kWays);async_producer=std::thread([this]{producer_loop();});}
            async_initialized=true;
            if(cpu_max)cpu_thread=std::thread([this]{cpu_loop();});
            async_thread=std::thread([this]{async_loop();});
            while(!async_started.load(std::memory_order_acquire))std::this_thread::yield();
        }

        // Legacy prefill/first-T1 warmup updates CLOCK after an earlier request. Refresh once before
        // any graph for this forward is queued; pointers and descriptor epochs stay unchanged.
        if(async_device_plan && async_map_dirty) {
            if(!drain_async() || !upload_device_map())return false;
            async_map_dirty=false;++async_map_reentries;
        }
        if(async_error() || !async_layers[(std::size_t)il])return false;
        *out=async_layers[(std::size_t)il]->group;return true;
    }
    bool drain_async() {
        if(!async_initialized)return true;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            if(async_error())return false;
            if(std::chrono::steady_clock::now()>deadline){async_fail(8,"service drain timeout");return false;}
            bool pending=async_active.load(std::memory_order_acquire);
            for(auto& a:async_layers)if(a && flag_load(&a->host->request)!=a->served.load(std::memory_order_acquire))pending=true;
            if(!pending)return cudaStreamSynchronize(copy)==cudaSuccess;
            std::this_thread::yield();
        }
    }
    void stop_async() {
        if(async_thread.joinable()) {
            if(!drain_async())async_fail(5,"async drain failed");
            async_stop.store(true,std::memory_order_release);
            // Wake GPU waiters on error/destruction before any mapped memory is released.
            for(auto& a:async_layers)if(a && a->host)flag_store(&a->host->error,async_failed.load()?async_failed.load():6);
            async_thread.join();
            cudaStreamSynchronize(copy);cudaDeviceSynchronize();
        }
        if(async_producer.joinable()) {
            {std::lock_guard<std::mutex> lock(producer_mutex);producer_stop=true;}
            producer_cv.notify_all();async_producer.join();
            std::fprintf(stderr,"HY4_FLASH_ASYNC_PARALLEL_STATS groups=%ld experts=%ld\n",producer_groups,producer_experts);
        }
        if(cpu_thread.joinable()) {
            async_stop.store(true,std::memory_order_release);
            {std::lock_guard<std::mutex> lock(cpu_mutex);cpu_stop=true;}
            cpu_cv.notify_one();cpu_thread.join();
        }
        if(cpu_copy){cudaStreamSynchronize(cpu_copy);cudaStreamDestroy(cpu_copy);cpu_copy=nullptr;}
        if(cpu_experts) {
            std::fprintf(stderr,"HY4_FLASH_CPU_STATS experts=%llu result_h2d_bytes=%llu compute_us=%llu\n",
                cpu_jobs.load(),cpu_result_bytes.load(),cpu_compute_us.load());
            // Report fixed and marginal CPU and GPU costs used by the split decision.
            if(cpu_split!=CpuSplit::Fixed)
                std::fprintf(stderr,"HY4_FLASH_CPU_SPLIT mode=%s samples=%llu solves=%llu dropped=%llu"
                    " ready=%d fit a=%.4f c=%.4f d=%.4f g0=%.4f g=%.4f p0=%.4f\n",
                    cpu_split==CpuSplit::Ratio?"ratio":"auto",split_samples,split_solves,split_fallbacks,
                    (int)cpu_cost.ready(),cpu_cost.a,cpu_cost.c,cpu_cost.d,cpu_cost.g0,cpu_cost.g,cpu_cost.p0);
            if(cpu_split==CpuSplit::Auto)
                std::fprintf(stderr,"HY4_FLASH_CPU_FIT mode=%s observed=%llu fitted=%d downweighted=%llu first_ms=%.3f first_weight=%.5f\n",
                    cpu_fit.name(),(unsigned long long)cpu_fit.observed,cpu_cost.ncpu,(unsigned long long)cpu_fit.downweighted,
                    cpu_fit.first_ms,cpu_fit.first_weight);
            if(cpu_probe.enabled)
                std::fprintf(stderr,"HY4_FLASH_CPU_EXPLORE mode=adaptive probes=%llu overpredicted=%llu\n",
                    (unsigned long long)cpu_probe.trials,(unsigned long long)cpu_probe.overpredicted);
            cpu_experts.reset();
        }
        // Descriptors outlive graph executables; release in ExpertSlices destructor.
    }

    bool wait_async_layer(int il,uint32_t expected) {
        if(il<0 || (std::size_t)il>=async_layers.size() || !async_layers[il])return false;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(flag_load(&async_layers[il]->host->done)<expected) {
            if(async_error())return false;
            if(std::chrono::steady_clock::now()>deadline){async_fail(9,"host layer completion timeout");return false;}
            std::this_thread::yield();
        }
        return !async_error();
    }

    #include "window_supply.inc"

    ~ExpertSlices() {
        stop_window();
        stop_async();
        if (exclusive && copy) cudaStreamSynchronize(copy); // also drains partial groups on failure
        for (auto stream : plane_streams) if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        if (planes_ready) cudaEventDestroy(planes_ready);
        if (pinned_ram) cudaFreeHost(pinned_ram);
        if (exchange) {
            cudaFreeHost(exchange);
            std::fprintf(stderr, "HY4_FLASH_RESIDENT swaps=%llu\n", (unsigned long long)exchanges);
        }
        // The prompt ring: device plane, three pinned buffers, an event and the slot map. Several hundred
        // MiB at a 64-slot ring, so leaking it once a session is not a rounding error.
        prefill_ring_close();
        async_layers.clear();
        if(async_device_slots)cudaFree(async_device_slots);
        if(async_epoch_arena)cudaFree(async_epoch_arena);
        if(async_device_plan)report_device_plan();
        if(prefill_admitted>0)std::fprintf(stderr,"HY4_FLASH_PREFILL_ADMITTED experts=%ld\n",prefill_admitted);
        if(hot_cache)std::fprintf(stderr,"HY4_FLASH_HOT_CACHE_COUNTERS admitted=%llu rejected=%llu hits=%ld misses=%ld evictions=%ld\n",
            hot_admitted,hot_rejected,hits,misses,evictions);
        for (int w = 0; w < kWays; ++w) {
            for (int r = 0; r < 3; ++r)
                if (pin[w][r] != nullptr) cudaFreeHost(pin[w][r]);
        }
        if (ev_copy[0]) cudaEventDestroy(ev_copy[0]);
        if (ev_arena) cudaEventDestroy(ev_arena);
        if (ev_group) cudaEventDestroy(ev_group);
        if (copy) cudaStreamDestroy(copy);
    }

    /// One routing block's three per-expert sizes, as the file gives them. `open_tier` groups blocks whose
    /// three agree into one pool, so a slot is sized for its own block.
    struct BlockSizes { int block; std::size_t per[3]; };

    /// Cap the expert tier at the available device budget. Zero disables caching.
    void open_tier(std::size_t budget, int n_block, int experts, int k,
                   const std::vector<BlockSizes>& sizes, std::size_t reserve_bytes = 0, Buf reuse = {}) {
        n_expert = experts;
        pool_of.assign((std::size_t) (n_block > 0 ? n_block : 0), -1);
        block_base.assign((std::size_t) (n_block > 0 ? n_block : 0), -1);
        if (n_block <= 0 || experts <= 0 || k <= 0 || sizes.empty()) return;
        // Store gate, up and down separately so plane strides are whole quantization blocks.
        const auto round_up = [](std::size_t n, std::size_t m) { return m == 0 ? n : (n + m - 1) / m * m; };
        // 144 covers Q4_K and 210 Q6_K; a size that is a whole number of neither rounds to the 5040 both
        // divide, which is the old behaviour and still correct, only wasteful.
        const auto align_for = [&](std::size_t bytes) -> std::size_t {
            if (bytes % 144 == 0) return 144;
            if (bytes % 210 == 0) return 210;
            return 5040;
        };
        // ---- group the routing blocks into pools by their three sizes
        for (const BlockSizes& b : sizes) {
            if (b.block < 0 || b.block >= n_block) continue;
            int at = -1;
            for (int p = 0; p < n_pools; ++p) {
                const TierPool& q = pools[p];
                if (q.stride_r[0] == round_up(b.per[0], align_for(b.per[0])) &&
                    q.stride_r[1] == round_up(b.per[1], align_for(b.per[1])) &&
                    q.stride_r[2] == round_up(b.per[2], align_for(b.per[2]))) {
                    at = p;
                    break;
                }
            }
            if (at < 0) {
                if (n_pools == kMaxPools) {
                    // More distinct shapes than pools. Fold this block into the widest pool: its slots are
                    // large enough for anything narrower, so this is the old padding behaviour for that
                    // block alone rather than a refusal.
                    int widest = 0;
                    for (int p = 1; p < n_pools; ++p)
                        if (pools[p].slot_bytes > pools[widest].slot_bytes) widest = p;
                    bool fits = true;
                    for (int r = 0; r < 3; ++r)
                        if (pools[widest].stride_r[r] < b.per[r]) fits = false;
                    if (!fits) continue;   // cannot hold it; that block keeps streaming
                    at = widest;
                } else {
                    at = n_pools++;
                    TierPool& q = pools[at];
                    for (int r = 0; r < 3; ++r) q.stride_r[r] = round_up(b.per[r], align_for(b.per[r]));
                    q.slot_bytes = q.stride_r[0] + q.stride_r[1] + q.stride_r[2];
                }
            }
            pool_of[(std::size_t) b.block] = at;
            // Each block gets a contiguous run of its pool's slots, so a full pool needs no plan at all:
            // block `il`'s expert `e` is always at `block_base[il] + e`.
            block_base[(std::size_t) b.block] = pools[at].n;
            pools[at].n += experts;   // every one of this block's experts may want a slot
        }
        if (n_pools == 0) return;
        // Even with a zero cache budget, the active group needs device staging slots.
        for (int p = 0; p < n_pools; ++p) pools[p].overflow = 2 * k;
        // What each pool would take to hold every expert of its blocks, kept so a rescale is always
        // computed from the full figure rather than from the last attempt.
        std::vector<int> full((std::size_t) n_pools);
        std::size_t want = 0;
        for (int p = 0; p < n_pools; ++p) {
            full[(std::size_t) p] = pools[p].n;
            want += (std::size_t) pools[p].n * pools[p].slot_bytes;
        }
        // Scale every pool by the same fraction of what it wanted, so a budget short of the whole trunk
        // still spreads across blocks instead of filling the first pool and starving the second.
        const auto rescale = [&](double frac) {
            for (int p = 0; p < n_pools; ++p) {
                TierPool& q = pools[p];
                int s = (int) ((double) full[(std::size_t) p] * frac);
                if (s > full[(std::size_t) p]) s = full[(std::size_t) p];
                if (s < q.overflow) s = q.overflow;
                q.n = s;
            }
        };
        const auto total = [&] {
            std::size_t b = 0;
            for (int p = 0; p < n_pools; ++p) b += (std::size_t) pools[p].n * pools[p].slot_bytes;
            return b;
        };
        const auto at_minimum = [&] {
            for (int p = 0; p < n_pools; ++p)
                if (pools[p].n > pools[p].overflow) return false;
            return true;
        };
        double f = want == 0 ? 0.0 : (double) budget / (double) want;
        if (f > 1.0) f = 1.0;
        rescale(f);
        // Ask for what is there rather than what was asked for; `room_for` keeps the pass its scratch.
        // Step down by a tenth, not by half: the full tier for this model is 85.5 GiB on a 95.6 GiB card,
        // so halving on the first refusal throws away 42 GiB that was there.
        reserve_bytes = std::min(reserve_bytes, want);
        while ((reuse.p ? total() > reserve_bytes : !room_for(std::max(total(), reserve_bytes))) && !at_minimum()) {
            f *= 0.9;
            rescale(f);
        }
        tier_capacity = std::max(total(), reserve_bytes);
        tier = reuse.p ? std::move(reuse) : Buf(tier_capacity);
        if (tier.p == nullptr) return;
        // A full tier needs no eviction or overflow slots.
        pools_full = true;
        for (int p = 0; p < n_pools; ++p)
            if (pools[p].n < full[(std::size_t) p]) pools_full = false;
        if (pools_full)
            for (int p = 0; p < n_pools; ++p) pools[p].overflow = 0;
        // ---- lay the planes out: pool by pool, role by role, each role's plane contiguous in its pool
        std::size_t at = 0;
        int g = 0;
        n_slots = 0;
        slot_bytes = 0;
        for (int p = 0; p < n_pools; ++p) {
            TierPool& q = pools[p];
            for (int r = 0; r < 3; ++r) {
                q.plane[r] = at;
                at += (std::size_t) q.n * q.stride_r[r];
            }
            q.g0 = g;
            g += q.n;
            n_slots += q.n;
            if (q.slot_bytes > slot_bytes) slot_bytes = q.slot_bytes;
        }
        slot_of.assign((std::size_t) n_block * (std::size_t) experts, -1);
        key_of.assign((std::size_t) n_slots, -1);
        used.assign((std::size_t) n_slots, 0);
        claimed.assign((std::size_t) n_slots, 0);
        const char* hot = std::getenv("STRATA_HY4_FLASH_HOT_CACHE");
        hot_cache = hot && std::atoi(hot) != 0;
        const char* cold_cpu = std::getenv("STRATA_HY4_FLASH_HOT_CPU_COLD_FIRST");
        hot_cpu_cold_first = hot_cache && cold_cpu && std::atoi(cold_cpu) != 0;
        if (hot_cache) {
            heat.reset(slot_of.size());
            std::fprintf(stderr,"HY4_FLASH_HOT_CPU_COLD_FIRST enabled=%d\n",(int)hot_cpu_cold_first);
            std::fprintf(stderr,"HY4_FLASH_HOT_CACHE enabled=1 sample_slots=%d decay_tokens=%llu online_only=1 decode_admission_only=1\n",
                ExpertHeat::sample_slots,(unsigned long long)ExpertHeat::decay_tokens);
        }
        idx_host.assign((std::size_t) k, 0);
        idx_dev = Buf((std::size_t) k * sizeof(int32_t));
        miss.reserve((std::size_t) k);
        // One thread per position in a group is the most parallelism a group can use; fewer when the
        // machine has fewer. 384 cores here, so this is about the group, not the machine.
        if (!pool) {
            const unsigned hw = std::thread::hardware_concurrency();
            const int n = (int) std::min<unsigned>(hw == 0 ? 8u : hw, (unsigned) kWays);
            pool.reset(new Pool(n));
        }
    }

    /// Initialize the copy stream and its events together.
    bool ensure_copy() {
        if (copy != nullptr) return true;
        if (cudaStreamCreate(&copy) != cudaSuccess) return false;
        for (cudaEvent_t* e : {&ev_arena, &ev_group, &ev_copy[0]})
            if (cudaEventCreateWithFlags(e, cudaEventDisableTiming) != cudaSuccess) return false;
        return true;
    }

    // Stage a prompt chunk's nonresident experts in a separate ring. Its capacity
    // bounds each MMQ batch without evicting experts from the decode cache.
    struct PrefillRing {
        void* dev = nullptr;              ///< [slots] x 3 planes of expert weights, device
        std::size_t plane[3] = {0, 0, 0}; ///< byte offset of each role's plane inside `dev`
        std::size_t stride_r[3] = {0, 0, 0};
        int slots = 0;
        void* pin[3] = {};                ///< one pinned staging buffer a role, `slots` experts wide
        std::size_t pin_bytes[3] = {0, 0, 0};
        cudaEvent_t filled{};             ///< the batch's H2D has landed
        bool have_event = false;
        cudaEvent_t used_event{};
        bool used_recorded = false;
        bool uploaded = false;
        std::vector<int32_t> slot_of_host; ///< [n_expert], ring slot or -1, rebuilt per batch
        Buf slot_dev;                      ///< its device copy
        /// Use the tier's device slot map for this block.
        Buf tier_slot_dev;
        int tier_slot_base = 0;
        int tier_slot_count = 0;
        int tier_slot_block = -1;          ///< which block `tier_slot_dev` holds
        bool tier_slot_dirty = true;
    };
    PrefillRing pring;
    PrefillRing pring_ahead;
    std::vector<int32_t> prefill_plan;
    std::size_t prefill_cursor = 0;
    int prefill_plan_block = -1;
    PrefillRing* prefill_current = nullptr;
    std::thread prefill_issuer;
    std::mutex prefill_issue_mutex;
    std::condition_variable prefill_issue_cv;
    std::function<void()> prefill_issue_job;
    std::exception_ptr prefill_issue_error;
    bool prefill_issue_stop = false, prefill_issue_pending = false;
    static bool prefill_host_async() {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_HOST_ASYNC");
            return v != nullptr && std::atoi(v) != 0;
        }();
        return on;
    }
    void prefill_issue_wait() {
        if (!prefill_host_async()) return;
        std::unique_lock<std::mutex> lock(prefill_issue_mutex);
        prefill_issue_cv.wait(lock, [&] { return !prefill_issue_pending; });
        if (prefill_issue_error) std::rethrow_exception(prefill_issue_error);
    }
    void prefill_issue_shutdown() {
        if (!prefill_issuer.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(prefill_issue_mutex);
            prefill_issue_stop = true;
        }
        prefill_issue_cv.notify_all(); prefill_issuer.join();
    }
    void prefill_schedule(std::size_t at, cudaStream_t stream) {
        if (!prefill_host_async()) { prefill_prepare(at, stream); return; }
        if (!prefill_issuer.joinable()) {
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("prefill issuer: no device");
            prefill_issue_stop = false;
            prefill_issuer = std::thread([this, device] {
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(prefill_issue_mutex);
                        prefill_issue_cv.wait(lock, [&] { return prefill_issue_stop || (bool) prefill_issue_job; });
                        if (!prefill_issue_job && prefill_issue_stop) return;
                        job = std::move(prefill_issue_job); prefill_issue_job = nullptr;
                    }
                    std::exception_ptr failure;
                    try {
                        if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("prefill issuer: set device failed");
                        job();
                    } catch (...) { failure = std::current_exception(); }
                    {
                        std::lock_guard<std::mutex> lock(prefill_issue_mutex);
                        prefill_issue_error = failure;
                        prefill_issue_pending = false;
                    }
                    prefill_issue_cv.notify_all();
                }
            });
        }
        {
            std::lock_guard<std::mutex> lock(prefill_issue_mutex);
            if (prefill_issue_pending) throw std::runtime_error("prefill issuer: outstanding supply job");
            if (prefill_issue_error) std::rethrow_exception(prefill_issue_error);
            prefill_issue_pending = true;
            prefill_issue_job = [this, at, stream] { prefill_prepare(at, stream); };
        }
        prefill_issue_cv.notify_all();
    }
    static bool prefill_pipeline_on() {
        static const bool on = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_PIPELINE");
            return v != nullptr && std::atoi(v) != 0;
        }();
        return on;
    }
    PrefillRing& prefill_bank(std::size_t at) {
        return ((at / (std::size_t) pring.slots) & 1) ? pring_ahead : pring;
    }

    /// Build the ring once. `slots` is the batch width; see the note above on what it costs and does not buy.
    bool prefill_ring_open_one(PrefillRing& pring, int slots, int n_expert_in, std::string& err) {
        if (pring.dev != nullptr) return true;
        if (slots <= 0 || n_expert_in <= 0 || pool_i < 0 || n_pools <= 0) {
            err = "prefill ring: no pool to size against";
            return false;
        }
        const TierPool& q = pools[pool_i];
        std::size_t total = 0;
        for (int r = 0; r < 3; ++r) {
            pring.stride_r[r] = q.stride_r[r];
            pring.plane[r] = total;
            total += (std::size_t) slots * q.stride_r[r];
        }
        // The MMQ tail guard, as everywhere else a sized slot is read by MMQ.
        total += strata::hy4_flash::mmq::kSlotGuard;
        if (cudaMalloc(&pring.dev, total) != cudaSuccess) {
            cudaGetLastError();
            err = "prefill ring: no room for " + std::to_string(total >> 20) + " MiB";
            return false;
        }
        if (cudaMemset(pring.dev, 0, total) != cudaSuccess) {   // the guard, and any slot never written
            cudaGetLastError();
            err = "prefill ring: cannot clear";
            return false;
        }
        for (int r = 0; r < 3; ++r) {
            pring.pin_bytes[r] = (std::size_t) slots * q.stride_r[r];
            if (cudaHostAlloc(&pring.pin[r], pring.pin_bytes[r], cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                err = "prefill ring: no pinned staging";
                return false;
            }
        }
        if (cudaEventCreateWithFlags(&pring.filled, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            err = "prefill ring: no event";
            return false;
        }
        pring.have_event = true;
        if (cudaEventCreateWithFlags(&pring.used_event, cudaEventDisableTiming) != cudaSuccess) {
            err = "prefill ring: no used event";
            return false;
        }
        pring.slots = slots;
        pring.slot_of_host.assign((std::size_t) n_expert_in, -1);
        pring.tier_slot_dev = Buf((std::size_t) n_expert_in * sizeof(int32_t));
        pring.slot_dev = Buf((std::size_t) n_expert_in * sizeof(int32_t));
        if (pring.slot_dev.p == nullptr || pring.tier_slot_dev.p == nullptr) {
            err = "prefill ring: no room for the slot map";
            return false;
        }
        return true;
    }

    bool prefill_ring_open(int total_slots, int n_expert_in, std::string& err) {
        // Two banks share the original slot budget; pipeline mode does not silently double VRAM.
        const bool pipe = prefill_pipeline_on();
        const int width = pipe ? total_slots / 2 : total_slots;
        if (width <= 0) { err = "prefill pipeline needs at least two total slots"; return false; }
        if (!prefill_ring_open_one(pring, width, n_expert_in, err) ||
            (pipe && !prefill_ring_open_one(pring_ahead, width, n_expert_in, err))) {
            prefill_ring_close(); // rollback partial allocation; never publish a half-built bank
            return false;
        }
        return true;
    }

    /// Stage a batch and publish its expert-to-ring-slot map after the copy completes.
    bool prefill_ring_stage(PrefillRing& pring, int block_id, const std::vector<int32_t>& want, const int32_t** slot_map_dev,
                            cudaStream_t compute, std::string& err, bool async = false) {
        if (pring.dev == nullptr || pring.slots <= 0) { err = "prefill ring: not open"; return false; }
        // Host staging cannot be reused until its previous DMA is complete. Device weights and map
        // cannot be overwritten until the last MMQ/admission consumer has released this bank.
        if (async) {
            if (pring.uploaded && cudaEventSynchronize(pring.filled) != cudaSuccess) {
                err = "prefill ring: previous DMA failed"; return false;
            }
            if (pring.used_recorded && cudaStreamWaitEvent(copy, pring.used_event, 0) != cudaSuccess) {
                err = "prefill ring: cannot wait for consumer"; return false;
            }
        } else if (cudaStreamSynchronize(compute) != cudaSuccess) {
            err = "prefill ring: cannot drain compute before reuse"; return false;
        }
        if ((int) want.size() > pring.slots) { err = "prefill ring: batch wider than the ring"; return false; }
        if (gg == nullptr || !pool) { err = "prefill ring: no file or pool"; return false; }
        for (int r = 0; r < 3; ++r) if (stride[r] > pring.stride_r[r]) {
            err = "prefill ring: later pool exceeds allocated plane stride"; return false;
        }

        // The map: -1 everywhere, then this batch's experts at their ring slot. Rebuilt rather than patched,
        // because a stale entry from the previous batch would silently read another expert's weights.
        std::fill(pring.slot_of_host.begin(), pring.slot_of_host.end(), -1);
        for (std::size_t i = 0; i < want.size(); ++i) {
            const int e = want[i];
            if (e < 0 || (std::size_t) e >= pring.slot_of_host.size()) {
                err = "prefill ring: expert id out of range";
                return false;
            }
            pring.slot_of_host[(std::size_t) e] = (int32_t) i;
        }

        const int n = (int) want.size();
        static const bool direct_on = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_DIRECT_DMA");
            return v != nullptr && std::atoi(v) != 0;
        }();
        bool direct = !exclusive && direct_on && gg->expert_pin_ready();
        std::vector<const char*> sources((std::size_t) n * 3);
        if (direct) for (int r = 0; r < 3; ++r) for (int i = 0; i < n; ++i) {
            sources[(std::size_t) r * n + i] = static_cast<const char*>(gg->at(
                base[r] + (uint64_t) want[(std::size_t) i] * stride[r], stride[r]));
            direct = direct && sources[(std::size_t) r * n + i] != nullptr;
        }
        if (exclusive && std::any_of(want.begin(), want.end(), [&](int e) {
                return slot_of[block_id * n_expert + e] >= 0;
            })) {
            const auto& q = pools[pool_of[block_id]];
            for (int r = 0; r < 3; ++r) for (int i = 0; i < n; ++i) {
                const int key = block_id * n_expert + want[i];
                const int slot = slot_of[key];
                const void* src = nullptr;
                if (slot >= 0) {
                    src = static_cast<const char*>(tier.p) + q.plane[r] + size_t(slot) * q.stride_r[r];
                } else {
                    src = ram_plane(key, r);
                    if (!src) { err = "prefill ring: missing RAM expert"; return false; }
                    if (!pinned_ram) {
                        auto* stage = static_cast<char*>(pring.pin[r]) + size_t(i) * pring.stride_r[r];
                        std::memcpy(stage, src, stride[r]);
                        src = stage;
                    }
                }
                auto* dst = static_cast<char*>(pring.dev) + pring.plane[r] + size_t(i) * pring.stride_r[r];
                if (cudaMemcpyAsync(dst, src, stride[r], slot >= 0 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                    copy) != cudaSuccess) {
                    err = "prefill ring: resident expert copy failed"; return false;
                }
            }
        } else if (direct) {
            OffloadNvtx mark("hy4-flash.prefill_direct_dma");
            for (int r = 0; r < 3; ++r) for (int i = 0; i < n;) {
                int end = i + 1;
                while (end < n && want[(std::size_t) end] == want[(std::size_t) end - 1] + 1 &&
                       sources[(std::size_t) r * n + end] == sources[(std::size_t) r * n + i] +
                           (std::size_t) (end - i) * stride[r]) ++end;
                char* dst = static_cast<char*>(pring.dev) + pring.plane[r] + (std::size_t) i * pring.stride_r[r];
                if (gg->copy_expert_rows_async(dst, pring.stride_r[r], sources[(std::size_t) r * n + i],
                                               stride[r], end - i, copy) != cudaSuccess) {
                    err = "prefill ring: direct H2D failed"; return false;
                }
                i = end;
            }
        } else {
        // Unregistered or transient sources retain bounded pinned staging.
        OffloadNvtx mark("hy4-flash.prefill_host_prepare");
        std::atomic<bool> failed{false};
        for (int r = 0; r < 3; ++r) {
            char* into = static_cast<char*>(pring.pin[r]);
            const std::size_t per = stride[r];
            pool->run(n, [&](int i) {
                if (failed.load()) return;
                // Compute source addresses from the plane's base and per-expert byte stride.
                const uint64_t from = base[r] + (uint64_t) want[(std::size_t) i] * per;
                char* dst = into + (std::size_t) i * pring.stride_r[r];
                if (!read_plane(*gg, block_id*n_expert + want[i], r, from, dst, per)) failed.store(true);
            });
            if (failed.load()) { err = "prefill ring: cannot read the expert weights"; return false; }
        }

        // ---- one H2D a role. The plane strides match the tier's, so MMQ reads the ring exactly as it
        // reads the tier: `w + slot * stride_r[r]`.
        for (int r = 0; r < 3; ++r) {
            char* to = static_cast<char*>(pring.dev) + pring.plane[r];
            if (cudaMemcpyAsync(to, pring.pin[r], (std::size_t) n * pring.stride_r[r],
                                cudaMemcpyHostToDevice, copy) != cudaSuccess) {
                cudaGetLastError();
                err = "prefill ring: H2D failed";
                return false;
            }
        }
        } // staged source
        if (cudaMemcpyAsync(pring.slot_dev.p, pring.slot_of_host.data(),
                            pring.slot_of_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                            copy) != cudaSuccess) {
            cudaGetLastError();
            err = "prefill ring: slot map H2D failed";
            return false;
        }
        if (cudaEventRecord(pring.filled, copy) != cudaSuccess ||
            (!async && cudaEventSynchronize(pring.filled) != cudaSuccess)) {
            cudaGetLastError();
            err = "prefill ring: cannot wait for the batch";
            return false;
        }
        pring.uploaded = true;
        prefill_weight_bytes.fetch_add((uint64_t)n*(stride[0]+stride[1]+stride[2]),std::memory_order_relaxed);
        prefill_weight_transfers.fetch_add(n,std::memory_order_relaxed);
        *slot_map_dev = static_cast<const int32_t*>(pring.slot_dev.p);
        return true;
    }

    /// Admit staged prompt experts into available permanent slots for subsequent chunks and decode.
    bool prefill_ring_admit(PrefillRing& bank, int block_id, const std::vector<int32_t>& want, cudaStream_t copy_on) {
        if (bank.dev == nullptr || pool_i < 0 || n_expert <= 0) return false;
        TierPool& q = pools[pool_i];
        const int permanent = q.n - q.overflow;
        bool any = false;
        for (std::size_t i = 0; i < want.size(); ++i) {
            const int e = want[i];
            if (e < 0 || e >= n_expert) continue;
            const std::size_t key = (std::size_t) block_id * (std::size_t) n_expert + (std::size_t) e;
            if (key >= slot_of.size() || slot_of[key] >= 0) continue;   // already resident
            if (q.next_free >= permanent) break;                        // pool full: leave it to decode
            const int slot = q.next_free++;
            for (int r = 0; r < 3; ++r) {
                const char* src = static_cast<const char*>(bank.dev) + bank.plane[r] +
                                  (std::size_t) i * bank.stride_r[r];
                char* dst = static_cast<char*>(tier.p) + q.plane[r] + (std::size_t) slot * q.stride_r[r];
                if (cudaMemcpyAsync(dst, src, q.stride_r[r], cudaMemcpyDeviceToDevice, copy_on) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    --q.next_free;        // the slot was never written; do not claim it
                    return any;
                }
            }
            slot_of[key] = slot;
            key_of[(std::size_t) (q.g0 + slot)] = (int32_t) key;
            used[(std::size_t) (q.g0 + slot)] = 1;
            async_map_dirty = true;       // the device copy of slot_of is now stale
            pring.tier_slot_dirty = true; // and so is the row the resident pass reads
            any = true;
            ++prefill_admitted;
        }
        return any;
    }

    long prefill_admitted = 0;   ///< experts the prompt put into free tier slots, cumulative
    std::atomic<uint64_t> prefill_weight_bytes{0},prefill_weight_transfers{0};

    /// The tier's `slot_of` row for one block, on the device. Re-uploaded when admission moved it or when
    /// the block changed; the row is `n_expert` ints and a chunk has 41 of them, so this is a few KB.
    const int32_t* prefill_tier_slots(int block_id, cudaStream_t on) {
        if (pring.tier_slot_dev.p == nullptr || n_expert <= 0) return nullptr;
        const std::size_t at = (std::size_t) block_id * (std::size_t) n_expert;
        if (at + (std::size_t) n_expert > slot_of.size()) return nullptr;
        if (pring.tier_slot_block != block_id || pring.tier_slot_dirty) {
            static const bool compact = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_RESIDENT_MMQ");
                return v != nullptr && std::atoi(v) != 0;
            }();
            const int32_t* upload = slot_of.data() + at;
            std::vector<int32_t> relative;
            pring.tier_slot_base = 0;
            pring.tier_slot_count = pools[pool_i].n;
            if (compact) {
                int lo = pools[pool_i].n, hi = -1;
                for (int e = 0; e < n_expert; ++e) if (upload[e] >= 0) {
                    lo = std::min(lo, upload[e]); hi = std::max(hi, upload[e]);
                }
                pring.tier_slot_base = hi >= 0 ? lo : 0;
                pring.tier_slot_count = hi >= 0 ? hi - lo + 1 : 0;
                // Match the staged MMQ grid to preserve its accumulation order.
                if (pring.tier_slot_count && pring.slots > 0) {
                    const int width = pring.slots;
                    const int rounded = ((pring.tier_slot_count + width - 1) / width) * width;
                    if (rounded <= pools[pool_i].n) {
                        pring.tier_slot_base = std::min(lo, pools[pool_i].n - rounded);
                        pring.tier_slot_count = rounded;
                    } else {
                        // The pool cannot supply a full grid; the staging ring can.
                        pring.tier_slot_count = 0;
                    }
                }
                relative.assign(upload, upload + n_expert);
                for (int32_t& v : relative) if (v >= 0) v -= pring.tier_slot_base;
                upload = relative.data();
            }
            // Restrict MMQ to this block's live physical span; otherwise most of its expert grid is
            // empty (the pool includes all layers). This also keeps the tile heuristic meaningful.
            if (cudaMemcpyAsync(pring.tier_slot_dev.p, upload,
                                (std::size_t) n_expert * sizeof(int32_t), cudaMemcpyHostToDevice,
                                on) != cudaSuccess) {
                cudaGetLastError();
                return nullptr;
            }
            if (cudaStreamSynchronize(on) != cudaSuccess) { cudaGetLastError(); return nullptr; }
            pring.tier_slot_block = block_id;
            pring.tier_slot_dirty = false;
        }
        return static_cast<const int32_t*>(pring.tier_slot_dev.p);
    }

    /// The ring as an `ExpertGroup`, so the batched MMQ addresses it the way it addresses the tier.
    void prefill_ring_group(const PrefillRing& pring, ExpertGroup* out) const {
        out->gate = static_cast<const char*>(pring.dev) + pring.plane[0];
        out->gate_stride = pring.stride_r[0];
        out->up = static_cast<const char*>(pring.dev) + pring.plane[1];
        out->up_stride = pring.stride_r[1];
        out->down = static_cast<const char*>(pring.dev) + pring.plane[2];
        out->down_stride = pring.stride_r[2];
        out->index = nullptr;
    }

    void prefill_ring_close_one(PrefillRing& pring) {
        if (pring.dev != nullptr) { cudaFree(pring.dev); pring.dev = nullptr; }
        for (int r = 0; r < 3; ++r)
            if (pring.pin[r] != nullptr) { cudaFreeHost(pring.pin[r]); pring.pin[r] = nullptr; }
        if (pring.have_event) { cudaEventDestroy(pring.filled); pring.have_event = false; }
        if (pring.used_event) { cudaEventDestroy(pring.used_event); pring.used_event = nullptr; }
        pring.used_recorded = false;
        pring.uploaded = false;
        pring.slot_dev = Buf();
        pring.tier_slot_dev = Buf();
        pring.tier_slot_block = -1;
        pring.slots = 0;
    }

    void prefill_ring_close() {
        prefill_issue_shutdown();
        // cudaFree waits for device users; host staging must also outlive outstanding H2D copies.
        if (copy != nullptr) cudaStreamSynchronize(copy);
        prefill_ring_close_one(pring);
        prefill_ring_close_one(pring_ahead);
        prefill_plan.clear();
        prefill_current = nullptr;
    }

    void prefill_plan_begin(int block_id, const int32_t* ids, int n, cudaStream_t stream) {
        if (n <= 0 || ids == nullptr) throw std::runtime_error("prefill pipeline: empty plan");
        prefill_issue_wait(); // previous plan/source metadata must not be changed under the issuer
        prefill_plan.assign(ids, ids + n);
        prefill_cursor = 0;
        prefill_plan_block = block_id;
        prefill_current = nullptr;
        prefill_schedule(0, stream);
    }

    void prefill_prepare(std::size_t at, cudaStream_t stream) {
        const std::size_t end = std::min(at + (std::size_t) pring.slots, prefill_plan.size());
        std::vector<int32_t> want(prefill_plan.begin() + at, prefill_plan.begin() + end);
        const int32_t* ignored = nullptr;
        std::string err;
        if (!prefill_ring_stage(prefill_bank(at), prefill_plan_block, want, &ignored, stream, err, true))
            throw std::runtime_error("prefill pipeline supply: " + err);
    }

    PrefillRing& prefill_acquire(int block_id, const std::vector<int32_t>& want, cudaStream_t stream) {
        if (block_id != prefill_plan_block || prefill_cursor + want.size() > prefill_plan.size() ||
            !std::equal(want.begin(), want.end(), prefill_plan.begin() + prefill_cursor))
            throw std::runtime_error("prefill pipeline: batch does not match published plan");
        prefill_issue_wait(); // ready event must have been RECORDED before the device wait is enqueued
        PrefillRing& bank = prefill_bank(prefill_cursor);
        if (cudaStreamWaitEvent(stream, bank.filled, 0) != cudaSuccess)
            throw std::runtime_error("prefill pipeline: cannot wait for ready bank");
        prefill_current = &bank;
        prefill_cursor += want.size();
        // The next bank is independent of this batch. Its used event was recorded after its previous
        // scatter/admission, so uploading it can overlap the current batch's three MMQ products.
        if (prefill_cursor < prefill_plan.size()) prefill_schedule(prefill_cursor, stream);
        return bank;
    }

    void prefill_release(cudaStream_t stream) {
        if (prefill_current == nullptr ||
            cudaEventRecord(prefill_current->used_event, stream) != cudaSuccess)
            throw std::runtime_error("prefill pipeline: cannot release consumed bank");
        prefill_current->used_recorded = true;
        prefill_current = nullptr;
    }

    bool prefill_all(Gguf& g, const std::vector<std::string>& prefix) {
        if (complete) return true;
        if (!pools_full || tier.p == nullptr || n_expert <= 0 || !pool) return false;
        if (prefix.size() < pool_of.size()) return false;
        if (!ensure_copy()) return false;
        // Use a pinned ring for asynchronous transfers from pageable model storage.
        static constexpr std::size_t kChunk = (std::size_t) 64 << 20;
        static constexpr int kRing = 2;
        void* stage[kRing] = {};
        cudaEvent_t done[kRing] = {};
        bool in_flight[kRing] = {};
        bool bad = false;
        for (int i = 0; i < kRing && !bad; ++i) {
            if (cudaHostAlloc(&stage[i], kChunk, cudaHostAllocDefault) != cudaSuccess) bad = true;
            else if (cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming) != cudaSuccess) bad = true;
        }
        const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        char* tbase = static_cast<char*>(tier.p);
        int ring = 0;
        for (int il = 0; il < (int) pool_of.size() && !bad; ++il) {
            const int p = pool_of[(std::size_t) il];
            if (p < 0 || block_base[(std::size_t) il] < 0) continue;
            const TierPool& q = pools[p];
            for (int r = 0; r < 3 && !bad; ++r) {
                const TensorInfo* t = g.info(prefix[(std::size_t) il] + names[r]);
                if (t == nullptr) { bad = true; break; }
                const uint64_t ne = t->shape.size() > 2 ? t->shape[2] : 1;
                if (ne == 0 || (int) ne != n_expert) { bad = true; break; }
                const std::size_t per = (std::size_t) (tensor_payload_bytes(*t) / ne);
                if (per == 0 || per > q.stride_r[r]) { bad = true; break; }
                const uint64_t src0 = g.data_start() + t->offset;
                char* dst0 = tbase + q.plane[r] +
                             (std::size_t) block_base[(std::size_t) il] * q.stride_r[r];
                // One run for the whole tensor when the plane's stride matches the file's - which is what
                // zero padding means - and expert by expert when it does not.
                const bool packed = per == q.stride_r[r];
                const int runs = packed ? 1 : n_expert;
                const std::size_t run = packed ? per * (std::size_t) n_expert : per;
                for (int u = 0; u < runs && !bad; ++u) {
                    for (std::size_t off = 0; off < run && !bad;) {
                        const std::size_t n = run - off < kChunk ? run - off : kChunk;
                        if (in_flight[ring] && cudaEventSynchronize(done[ring]) != cudaSuccess) {
                            bad = true;
                            break;
                        }
                        const uint64_t from = src0 + (uint64_t) u * per + off;
                        char* to = dst0 + (std::size_t) u * q.stride_r[r] + off;
                        const int parts =
                            (int) (n >> 20 == 0 ? 1 : (n >> 20 < kWays ? n >> 20 : (std::size_t) kWays));
                        std::atomic<bool> failed{false};
                        char* into = static_cast<char*>(stage[ring]);
                        pool->run(parts, [&](int i) {
                            const std::size_t lo = n * (std::size_t) i / (std::size_t) parts;
                            const std::size_t hi = n * (std::size_t) (i + 1) / (std::size_t) parts;
                            if (hi <= lo) return;
                            if (const void* src = exclusive ? nullptr : g.at(from + lo, hi - lo))
                                std::memcpy(into + lo, src, hi - lo);
                            else if (!g.pread_at(from + lo, into + lo, hi - lo))
                                failed.store(true);
                        });
                        if (failed.load()) { bad = true; break; }
                        if (cudaMemcpyAsync(to, stage[ring], n, cudaMemcpyHostToDevice, copy) !=
                            cudaSuccess) {
                            bad = true;
                            break;
                        }
                        cudaEventRecord(done[ring], copy);
                        in_flight[ring] = true;
                        ring = (ring + 1) % kRing;
                        h2d_bytes += n;
                        off += n;
                    }
                }
            }
        }
        if (!bad && cudaStreamSynchronize(copy) != cudaSuccess) bad = true;
        for (int i = 0; i < kRing; ++i) {
            if (done[i] != nullptr) cudaEventDestroy(done[i]);
            if (stage[i] != nullptr) cudaFreeHost(stage[i]);
        }
        if (bad) return false;
        // The residency table every block now satisfies, and its device copy. Both are constant from here:
        // nothing is evicted, because nothing has to be.
        for (int il = 0; il < (int) pool_of.size(); ++il) {
            if (pool_of[(std::size_t) il] < 0) continue;
            const std::size_t at = (std::size_t) il * (std::size_t) n_expert;
            for (int e = 0; e < n_expert; ++e)
                slot_of[at + (std::size_t) e] = block_base[(std::size_t) il] + e;
        }
        slot_dev = Buf(slot_of.size() * sizeof(int32_t));
        if (slot_dev.p == nullptr) return false;
        if (cudaMemcpy(slot_dev.p, slot_of.data(), slot_of.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
        for (int p = 0; p < n_pools; ++p) pools[p].next_free = pools[p].n;
        complete = true;
        return true;
    }

    /// Locate this block's three resident expert planes and their strides.
    bool resident_group_raw(int il, ExpertGroup* out) const {
        if (il < 0 || (std::size_t) il >= pool_of.size()) return false;
        const int p = pool_of[(std::size_t) il];
        if (p < 0 || tier.p == nullptr) return false;
        const TierPool& q = pools[p];
        out->gate = static_cast<const char*>(tier.p) + q.plane[0];
        out->gate_stride = q.stride_r[0];
        out->up = static_cast<const char*>(tier.p) + q.plane[1];
        out->up_stride = q.stride_r[1];
        out->down = static_cast<const char*>(tier.p) + q.plane[2];
        out->down_stride = q.stride_r[2];
        out->index = nullptr;
        return true;
    }

    bool resident_group(int il, ExpertGroup* out) const {
        if (!complete || il < 0 || (std::size_t) il >= pool_of.size()) return false;
        const int p = pool_of[(std::size_t) il];
        if (p < 0) return false;
        const TierPool& q = pools[p];
        out->gate = static_cast<const char*>(tier.p) + q.plane[0];
        out->gate_stride = q.stride_r[0];
        out->up = static_cast<const char*>(tier.p) + q.plane[1];
        out->up_stride = q.stride_r[1];
        out->down = static_cast<const char*>(tier.p) + q.plane[2];
        out->down_stride = q.stride_r[2];
        out->index = nullptr;   // the caller maps the router's ids through `slot_dev`
        return true;
    }

    /// Point at one block's three expert tensors. False when any is missing or will not fit.
    bool open_block(Gguf& g, const std::string& prefix, int block_index, int expert_dim, int expert_ff,
                    cudaStream_t compute) {
        gg = &g;
        ok = false;
        block = block_index;
        // Which pool this block's slots come from, decided once in `open_tier` from the file's own sizes.
        pool_i = block_index >= 0 && (std::size_t) block_index < pool_of.size()
                     ? pool_of[(std::size_t) block_index] : -1;
        if (pool_i < 0) return false;
        if (!ensure_copy()) return false;
        (void) compute;   // the arena barrier is per token, in `begin`
        const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (int r = 0; r < 3; ++r) {
            const TensorInfo* t = g.info(prefix + names[r]);
            if (t == nullptr) return false;
            base[r] = g.data_start() + t->offset;
            // ne2 is the expert count, so one expert is the payload divided by it.
            const uint64_t n_expert = t->shape.size() > 2 ? t->shape[2] : 1;
            if (n_expert == 0) return false;
            stride[r] = (std::size_t) (tensor_payload_bytes(*t) / n_expert);
            for (int w = 0; w < kWays; ++w) {
                // Size slots by bytes because expert quantization can vary by layer.
                if (pin_bytes[w][r] < stride[r]) {
                    // Nothing may be reading it. The copy stream is the only other user of this way and
                    // its last operation on it is `ev_copy[w]`.
                    if (ev_copy_set[0]) cudaEventSynchronize(ev_copy[0]);
                    if (pin[w][r] != nullptr) cudaFreeHost(pin[w][r]);
                    pin[w][r] = nullptr;
                    pin_bytes[w][r] = 0;
                    if (cudaHostAlloc(&pin[w][r], stride[r], cudaHostAllocDefault) != cudaSuccess) return false;
                    pin_bytes[w][r] = stride[r];
                }
            }
        }
        (void) expert_dim;
        (void) expert_ff;
        // The no-mapping fallback needs its own slice per position, since the positions read at once.
        plane_h[0] = 0;
        plane_h[1] = stride[0];
        plane_h[2] = stride[0] + stride[1];
        host_stride = stride[0] + stride[1] + stride[2];
        if (!g.mapped() && host.size() < (std::size_t) kWays * host_stride)
            host.resize((std::size_t) kWays * host_stride);
        ok = true;
        return true;
    }

    /// Resolve cache hits and stage misses before exposing the group's slot indices.
    bool fetch_group(const int32_t* ids, int k, ExpertGroup* out, cudaStream_t compute) {
        async_map_dirty=true; // legacy CPU planner; next async entry must refresh the stable GPU map.
        if (!ok || pool_i < 0 || pools[pool_i].n <= 0 || k > (int) idx_host.size()) return false;
        cudaEventRecord(ev_arena, compute);
        cudaStreamWaitEvent(copy, ev_arena, 0);

        OffloadNvtx fetch_nvtx("hy4-flash.fetch_group");
        // Plan all hits and staging destinations using pool-local indices.
        TierPool& q = pools[pool_i];
        const int g0 = q.g0;
        const int permanent = q.n - q.overflow;
        { OffloadNvtx plan_nvtx("hy4-flash.cache_plan");
        std::vector<uint8_t> hit_mask;
        if(!plan_slots(block,pool_i,ids,k,idx_host,hit_mask,miss))return false;

        } // cache plan
        static const bool guard_slots = std::getenv("HY4_FLASH_CAUSAL_SLOT_GUARD") != nullptr && std::string(std::getenv("HY4_FLASH_CAUSAL_SLOT_GUARD")) == "1";
        if (guard_slots) {
            OffloadNvtx guard_nvtx("hy4-flash.slot_guard");
            for (int j = 0; j < k; ++j) {
                const int s = idx_host[(std::size_t)j];
                const int key = block * n_expert + ids[j];
                bool valid = s >= 0 && s < q.n;
                if (valid && s < permanent) valid = key_of[(std::size_t)(g0+s)] == key;
                else if (valid) {
                    valid = false;
                    for (const Miss& m : miss) if (m.slot == s && m.expert == ids[j]) valid = true;
                }
                if (!valid) {
                    std::fprintf(stderr, "CAUSAL_WRONG_SLOT block=%d selected=%d expert=%d slot=%d\n",block,j,ids[j],s);
                    return false;
                }
            }
        }

        // Wait for the previous copies before reusing their pinned buffers.
        if (exclusive && !miss.empty()) {
            if (!transfer_exclusive(block, stride, miss)) return false;
        } else if (!miss.empty()) {
            static const bool direct_option=[] {
                const char* v=std::getenv("STRATA_HY4_FLASH_VERIFY_DIRECT_DMA");return v&&std::atoi(v)!=0;
            }();
            const bool direct=direct_option && gg->expert_pin_ready();
            { OffloadNvtx wait_nvtx("hy4-flash.previous_copy_wait");
            if (ev_copy_set[0]) cudaEventSynchronize(ev_copy[0]); }
            const auto t0 = std::chrono::steady_clock::now();
            bool bad = false;
            { OffloadNvtx stage_nvtx("hy4-flash.mmap_to_pinned");
            if (!direct) pool->run((int) miss.size(), [&](int i) {
                const Miss& m = miss[(std::size_t) i];
                for (int r = 0; r < 3; ++r) {
                    const std::size_t n = stride[r];
                    const uint64_t off = base[r] + (uint64_t) m.expert * n;
                    const void* src = gg->at(off, n);
                    if (src == nullptr) {
                        uint8_t* mine = host.data() + (std::size_t) m.pin_slot * host_stride + plane_h[r];
                        if (!gg->pread_at(off, mine, n)) {
                            bad = true;
                            return;
                        }
                        src = mine;
                    }
                    std::memcpy(static_cast<char*>(pin[m.pin_slot][r]), src, n);
                }
            });
            } // pinned staging
            if (bad) return false;
            t_read += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            OffloadNvtx upload_nvtx("hy4-flash.enqueue_expert_h2d");
            char* tbase = static_cast<char*>(tier.p);
            for (const Miss& m : miss) {
                for (int r = 0; r < 3; ++r) {
                    const std::size_t n = stride[r];
                    if (gg->copy_expert_async(tbase + q.plane[r] + (std::size_t) m.slot * q.stride_r[r],
                                              direct ? gg->at(base[r]+(uint64_t)m.expert*n,n) : pin[m.pin_slot][r],
                                              n, copy) != cudaSuccess)
                        return false;
                    h2d_bytes += n;
                }
            }
            cudaEventRecord(ev_copy[0], copy);
            ev_copy_set[0] = true;
            n_fetch += (long) miss.size();
        }

        // The index itself, and the one wait the arithmetic needs: everything staged above is on the copy
        // stream, so one event covers all of it.
        OffloadNvtx idx_nvtx("hy4-flash.index_h2d_and_compute_wait");
        if (cudaMemcpyAsync(idx_dev.p, idx_host.data(), (std::size_t) k * sizeof(int32_t),
                            cudaMemcpyHostToDevice, copy) != cudaSuccess)
            return false;
        cudaEventRecord(ev_group, copy);
        cudaStreamWaitEvent(compute, ev_group, 0);

        out->gate = static_cast<const char*>(tier.p) + q.plane[0];
        out->gate_stride = q.stride_r[0];
        out->up = static_cast<const char*>(tier.p) + q.plane[1];
        out->up_stride = q.stride_r[1];
        out->down = static_cast<const char*>(tier.p) + q.plane[2];
        out->down_stride = q.stride_r[2];
        out->index = idx_dev.as<int32_t>();
        return true;
    }
};

/// A scratch slot for this pass: a raw pointer into an allocation that outlives the call.
struct S {
    void* p = nullptr;
    template <class T>
    T* as() const { return static_cast<T*>(p); }
};

/// Reuse device scratch by slot and grow it only when a pass needs more capacity.
struct Scratch {
    enum Slot {
        kPos, kKvPos, kState, kNext, kEmbedSc, kHeadSc, kHcSc, kMixes, kPre, kPost, kReduced,
        kAttnOut, kFfnOut, kLogits, kEngramRows, kEngramValid, kEngramSc, kLen,
        kProjSc, kAbsorbSc, kAbsorbed, kAttnSc, kKqv, kOutSc, kFfnSc, kCurQ8,
        kIndexerSc, kIndexerSel, kExactMeta, kCount
    };
    Buf buf[kCount];
    std::size_t cap[kCount] = {};
    uint64_t generation = 0;   // changes whenever an address can change
    bool borrowed_layout = false;
    std::size_t borrowed_used[kCount] = {};
    void clear() {
        for (int i=0;i<kCount;++i) { buf[i].reset(); cap[i]=0; borrowed_used[i]=0; }
        borrowed_layout=false; ++generation;
    }
    void lend(void* base, std::size_t bytes, const std::array<std::size_t,kCount>& sizes) {
        clear(); std::size_t at=0;
        for (int i=0;i<kCount;++i) {
            const std::size_t n=(sizes[i]+255)&~std::size_t(255);
            if (n > bytes-at) throw std::runtime_error("prefill loan: scratch layout exceeds borrowed region");
            if (n) {buf[i]=Buf::borrowed(static_cast<char*>(base)+at);cap[i]=sizes[i];}
            at+=n;
        }
        borrowed_layout=true;
    }

    /// `bytes` of slot `slot`. Empty on failure, which every caller already checks for.
    S get(int slot, std::size_t bytes) {
        if (slot < 0 || slot >= kCount) return S{};
        if (bytes == 0) return S{};
        if(borrowed_layout && bytes<=cap[slot] && bytes>borrowed_used[slot]){
            // Ordinary growth allocates a zeroed Buf. A preplanned maximum capacity must not skip
            // that initialization when a later layer first requests a larger logical shape.
            if(cudaMemset(buf[slot].p,0,bytes)!=cudaSuccess)throw std::runtime_error("prefill loan: scratch initialization failed");
            borrowed_used[slot]=bytes;
        }
        // Re-made when it is too small, and also when poisoning is on - or the NaN diagnostic would only
        // ever have covered a slot's first pass.
        if (borrowed_layout && cap[slot] < bytes)
            throw std::runtime_error("prefill loan: scratch slot "+std::to_string(slot)+" needs "+
                std::to_string(bytes)+" but planned "+std::to_string(cap[slot]));
        // Release the old scratch before allocating a larger replacement to avoid a transient double
        // allocation.
        if (cap[slot] < bytes || (g_poison && !borrowed_layout)) {
            ++generation;
            buf[slot].reset();
            cap[slot] = 0;
            buf[slot] = Buf(bytes);
            cap[slot] = buf[slot].p != nullptr ? bytes : 0;
        }
        S s;
        s.p = buf[slot].p;
        return s;
    }
};

std::array<std::size_t,Scratch::kCount> prefill_workspace_plan(Gguf& gg, const Geometry& g,
        int nt, int kv_stride, int base_pos, int head_rows, int mmq_slots, bool full_rows, int logit_rows) {
    std::array<std::size_t,Scratch::kCount> z{};
    const auto take=[&](int id,std::size_t n){z[id]=std::max(z[id],n);};
    const int d=g.hidden_dim,hc=g.hc_mult,kr=g.kv_lora_rank,qr=g.q_lora_rank,rot=g.rope_dim;
    take(Scratch::kPos,nt*4ull); take(Scratch::kKvPos,kv_stride*4ull);take(Scratch::kLen,8);
    for(int id:{Scratch::kState,Scratch::kNext})take(id,4ull*d*hc*nt);
    take(Scratch::kEmbedSc,embed_tokens_scratch_bytes(d,nt));
    take(Scratch::kHcSc,hc_scratch_bytes(d*hc,nt));take(Scratch::kMixes,8ull*hc*nt);
    take(Scratch::kPre,4ull*hc*nt);take(Scratch::kPost,4ull*hc*nt);
    for(int id:{Scratch::kReduced,Scratch::kAttnOut,Scratch::kFfnOut})take(id,4ull*d*nt);
    take(Scratch::kCurQ8,kernels::native_q8_1_bytes_wide(d,nt));
    take(Scratch::kEngramRows,4ull*g.engram_heads*g.engram_head_dim*nt);
    take(Scratch::kEngramValid,4ull*g.engram_heads*nt);
    take(Scratch::kEngramSc,engram_layer_scratch_bytes(d,hc,g.engram_head_dim,g.engram_heads,nt));
    take(Scratch::kLogits,4ull*g.vocab_size*logit_rows);
    take(Scratch::kHeadSc,output_head_scratch_bytes(d,hc,g.vocab_size,head_rows));
    const auto info=[&](int il,const char* suffix){const auto* t=gg.info(blk(il)+suffix);
        if(!t)throw std::runtime_error("prefill loan: missing tensor "+blk(il)+suffix);return t;};
    const int nb=(int)g.layers.size()-g.n_mtp_blocks;
    for(int il=0;il<nb;++il){
        const auto& l=g.layers[il];const int h=l.n_heads;
        const int qh=(int)info(il,"attn_q_b.weight")->shape[1]/h;
        const int vh=(int)info(il,"attn_v_b.weight")->shape[1];
        const int sel=l.attention==Attention::Sliding?0:g.idx_top_k+g.idx_kpool;
        take(Scratch::kProjSc,mla_project_scratch_bytes(d,qr,kr,qh,rot,h,nt));
        take(Scratch::kAbsorbSc,mla_absorb_scratch_bytes(qh-rot,nt,h));
        take(Scratch::kAbsorbed,4ull*kr*nt*h);
        take(Scratch::kAttnSc,mla_attention_scratch_bytes(kr,vh,h,nt,kv_stride,sel));
        take(Scratch::kKqv,4ull*vh*h*nt);take(Scratch::kOutSc,mla_output_scratch_bytes(d,vh*h,nt));
        if(sel>0 && g.idx_kpool>0){
            const int max_nsel=g.idx_top_k+g.idx_kpool;
            const int mp=(kv_stride+g.idx_kpool-1)/g.idx_kpool+1;
            take(Scratch::kIndexerSc,indexer_scratch_bytes(g.idx_heads,g.idx_key_dim,g.idx_kpool,nt,mp,max_nsel));
            take(Scratch::kIndexerSel,4ull*std::max(max_nsel,1)*nt);
        }
        if(l.expert_ordinal<0)take(Scratch::kFfnSc,dense_ffn_scratch_bytes(d,(int)info(il,"ffn_gate.weight")->shape[1],nt));
        else {
            const auto* ge=info(il,"ffn_gate_exps.weight");const auto* ue=info(il,"ffn_up_exps.weight");
            const auto* de=info(il,"ffn_down_exps.weight");
            const int ed=(int)info(il,"ffn_routed_down.weight")->shape[1],ef=(int)ge->shape[0];
            const int ne=(int)info(il,"ffn_gate_inp.weight")->shape[1];
            take(Scratch::kFfnSc,latent_moe_scratch_bytes(d,ed,ef,ne,g.top_k,nt,(int)ge->type,
                (int)ue->type,(int)de->type,std::max(ne,mmq_slots),(full_rows?0:moe_group_rows(g.top_k,nt))));
        }
    }
    return z;
}

bool fetch_experts_cb(const int32_t* ids, int k, ExpertGroup* out, void* stream, void* user) {
    return static_cast<ExpertSlices*>(user)->fetch_group(ids, k, out, static_cast<cudaStream_t>(stream));
}

/// Supply missing experts from the prompt ring, preserving the decode tier.
bool stage_experts_cb(int block_id, const int32_t* ids, int n, ExpertGroup* out,
                      const int32_t** slot_of_dev, void* stream, void* user) {
    auto* s = static_cast<ExpertSlices*>(user);
    std::string err;
    std::vector<int32_t> want(ids, ids + n);
    ExpertSlices::PrefillRing* bank = &s->pring;
    if (s->prefill_pipeline_on()) {
        bank = &s->prefill_acquire(block_id, want, static_cast<cudaStream_t>(stream));
        *slot_of_dev = static_cast<const int32_t*>(bank->slot_dev.p);
    } else if (!s->prefill_ring_stage(*bank, block_id, want, slot_of_dev,
                                     static_cast<cudaStream_t>(stream), err)) {
        throw std::runtime_error("prefill supply: " + err);
    }
    s->prefill_ring_group(*bank, out);
    // Fill free tier slots from what we just staged, so the next chunk and the first decode steps find
    // these experts resident. Opt-out with STRATA_HY4_FLASH_PREFILL_ADMIT=0 to measure what it is worth.
    static const bool admit = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_ADMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (admit) s->prefill_ring_admit(*bank, block_id, want, static_cast<cudaStream_t>(stream));
    return true;
}

void stage_plan_cb(int block, const int32_t* ids, int n, void* stream, void* user) {
    static_cast<ExpertSlices*>(user)->prefill_plan_begin(block, ids, n, static_cast<cudaStream_t>(stream));
}
void stage_release_cb(void* stream, void* user) {
    static_cast<ExpertSlices*>(user)->prefill_release(static_cast<cudaStream_t>(stream));
}


}  // namespace

/// What the residency holds for one block (or, in the last slot, for the embedding and head): the device
/// copy of each tensor, by name, with the type and shape the file gave so a later pass need not re-read the
/// directory either.
struct DenseResidency::Slot {
    struct Entry {
        void* p = nullptr;   ///< nullptr both for "absent from the file" and "would not fit"; the callers
                             ///< already treat a null weight as absent, so the two need not be told apart
        bool owns_pointer = true; // packed entries borrow their Gguf owner
        int type = 0;
        std::vector<int64_t> shape;
    };
    std::unordered_map<std::string, Entry> by_name;
    ~Slot() {
        for (auto& kv : by_name)
            if (kv.second.p != nullptr && kv.second.owns_pointer) cudaFree(kv.second.p);
    }
};

/// The file, kept open and mapped for the residency's life. See DenseResidency::Open in the header for why
/// that is not an optimisation but a correctness condition for the warm page cache.
struct DenseResidency::Open {
    Gguf gg;
    explicit Open(const std::string& path) : gg(path) {}
};

/// What a pass needs that is neither a weight nor the file: its device scratch and its expert fetcher.
/// Held across calls for the same reason as everything else here - allocating it per token was the cost.
struct DenseResidency::Work {
    bool async_primed=false;
    bool single_phase=false;
    std::vector<Buf> async_engram_rows,async_engram_valid;
    std::vector<std::size_t> async_engram_rows_cap,async_engram_valid_cap;
    Scratch sc;
    ExpertSlices slices;
    int borrow_phase = -1;  // -1: never configured, 0: decode, 1: prefill
    int borrow_tokens = 0;
    bool borrow_failed = false;
    Buf prefill_device_logits;  // GPU sampler consumes these after returning the loan
    int auto_prompt_chunk = 0; // fixed until this prompt commits its final logits
    std::vector<int32_t> decode_slot_of; // pre-prompt permanent residency, including original slot numbers
    std::vector<ExpertSlices::TierPool> decode_pools;
    std::vector<uint8_t> decode_used; // CLOCK second-chance bits for original permanent slots
    std::vector<uint8_t> borrowed_observed;
    void* borrowed_layer = nullptr;
    std::size_t borrowed_layer_bytes = 0;

    /// The Engram row reader, when `--engram-io direct` asked for one. HERE and not in the pass, because it
    /// owns a file handle, an I/O worker and a row cache and none of them pay for themselves inside a call.
    std::unique_ptr<EngramRows> engram;
    std::vector<uint8_t> engram_raw;   ///< the reader's destination, one step's rows, reused
    /// Keep row-reader threads alive across decode steps.
    std::unique_ptr<Pool> engram_pool;
    /// One captured attention half per trunk block, built on the first decode step and replayed after.
    std::vector<cudaGraphExec_t> attn_graph;
    /// Graphs embed scratch addresses. Invalidate them when the captured layout changes.
    int graph_nt = 0;
    int captured_nt() const { return graph_nt > 0 ? graph_nt : 1; }
    /// Warm each verification width before capturing its graph.
    int spec_warm_nt = 0;
    /// Keep pooled sparse-indexer keys for the lifetime of the matching KV cache.
    std::unique_ptr<IndexerCache> idx;
    bool idx_engaged = false;   ///< last pass's `plan.engaged`; a flip rebuilds every graph
    // Opt-in T1 trunk/Engram/head graph. Embedding and all host input preparation remain outside.
    cudaGraphExec_t token_graph = nullptr;
    struct WindowGraph { cudaGraphExec_t exec=nullptr; cudaGraph_t source=nullptr; bool warm=false; };
    WindowGraph window_graph[9][4];
    Buf window_hidden;
    uint64_t window_generation=0;
    int window_kv_stride=0;
    int window_kv_format=-1;
    std::vector<void*> window_kv,window_pe;
    void invalidate_window_graphs() {
        for(auto& row:window_graph)for(auto& graph:row) {
            if(graph.exec)cudaGraphExecDestroy(graph.exec);
            if(graph.source)cudaGraphDestroy(graph.source);
            graph.exec=nullptr;graph.source=nullptr;graph.warm=false;
        }
    }
    bool token_warm = false;
    uint64_t token_scratch_generation = 0;
    int token_kv_stride = 0;
    int token_kv_format = -1;
    std::vector<void*> token_kv, token_pe;
    void invalidate_token_graph() {
        if (token_graph) { cudaGraphExecDestroy(token_graph); token_graph = nullptr; }
        token_warm = false;
    }
    // One entry per (width, engaged). All entries use the persistent scratch and are invalidated
    // together when any captured allocation changes. A real eager pass warms each new width first.
    struct VerifyGraph { cudaGraphExec_t exec = nullptr; bool warm = false, replay_reported = false; };
    std::array<VerifyGraph, (DflyDrafter::kMaxK + 1) * 2> verify_graphs{};
    uint64_t verify_scratch_generation = 0;
    std::vector<const void*> verify_addresses;
    std::vector<int> verify_shape;
    void invalidate_verify_graphs() {
        for (auto& g : verify_graphs) {
            if (g.exec) cudaGraphExecDestroy(g.exec);
            g = {};
        }
    }
    /// Capture only after dense uploads finish and device pointers are stable.
    bool warm = false;
    ~Work() {
        slices.stop_window();
        slices.stop_async();
        invalidate_window_graphs();
        invalidate_token_graph();
        invalidate_verify_graphs();
        for (cudaGraphExec_t g : attn_graph)
            if (g != nullptr) cudaGraphExecDestroy(g);
    }
};

DenseResidency::~DenseResidency() {
    delete work;
    delete file;
    for (Slot* s : slots) delete s;
}

namespace {

/// One weight for the duration of a block: the residency's device copy when there is one, otherwise a
/// buffer owned here and freed at the end of the pass, as before.
struct W {
    Buf own;             ///< non-empty only when there is no residency to hold this
    void* p = nullptr;
    int type = 0;
    std::vector<int64_t> shape;   ///< copied, not pointed at: four int64s, and a pointer into the
                                  ///< residency would be one more thing to keep valid
    template <class T>
    T* as() const { return static_cast<T*>(p); }
    int64_t dim(int i) const { return (int) shape.size() > i ? shape[(std::size_t) i] : 0; }
    /// Frees only what this pass owns. A resident weight is not ours to free - the point of it is that the
    /// next step finds it still there.
    void reset() {
        if (own.p != nullptr) {
            own.reset();
            p = nullptr;
        }
    }
};

/// Reuse the resident tensor when available, otherwise upload it from the model.
W take(Gguf& gg, DenseResidency::Slot* r, std::size_t* bytes, const std::string& name) {
    W w;
    if (gg.dense_arena_ready()) gg.dense_take_check(name);
    if (r == nullptr) {
        w.own = gg.upload(name, &w.type, &w.shape);
        w.p = w.own.p;
        return w;
    }
    auto it = r->by_name.find(name);
    if (it == r->by_name.end()) {
        DenseResidency::Slot::Entry e;
        if (gg.dense_arena_ready()) {
            e.p = gg.dense_pointer(name);
            e.owns_pointer = false;
            if (const TensorInfo* t = gg.info(name)) {
                e.type = (int)t->type;
                for (uint64_t d : t->shape) e.shape.push_back((int64_t)d);
                if (e.p != nullptr && bytes != nullptr) *bytes += (std::size_t)tensor_payload_bytes(*t);
            }
        } else {
            Buf b = gg.upload(name, &e.type, &e.shape);
            if (b.p != nullptr && bytes != nullptr) {
                const TensorInfo* t = gg.info(name);
                if (t != nullptr) *bytes += (std::size_t) tensor_payload_bytes(*t);
            }
            e.p = b.release();
        }
        it = r->by_name.emplace(name, std::move(e)).first;
    }
    w.p = it->second.p;
    w.type = it->second.type;
    w.shape = it->second.shape;
    return w;
}

/// Touch mapped pages sequentially to warm the file cache without allocating a second copy.
std::size_t warm_range(const void* p, std::size_t bytes) {
    if (p == nullptr || bytes == 0) return 0;
#ifndef _WIN32
    (void) ::madvise(const_cast<void*>(p), bytes, MADV_WILLNEED);
#endif
    const volatile uint8_t* q = static_cast<const volatile uint8_t*>(p);
    std::size_t page = 4096;
    uint8_t sink = 0;
    for (std::size_t at = 0; at < bytes; at += page) sink ^= q[at];
    sink ^= q[bytes - 1]; // an unaligned tensor can end on one more page
    (void) sink;
    return bytes;
}


}  // namespace

/// Limit prefill width independently of the single-launch matvec limit.
int prefill_compute_chunk() {
    if(g_auto_chunk>0)return g_auto_chunk;
    static const int n = [] {
        int chunk = 512;
        if (const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_CHUNK")) {
            const int want = std::atoi(v);
            if (want >= 1) chunk = want;
        }
        return chunk;
    }();
    return n;
}

int prefill_chunk_max() {
    if(auto_chunk_on())return auto_chunk_cap();
    const int chunk=prefill_compute_chunk();
    if(!layer_major_on())return chunk;
    const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_SUPERCHUNK");
    const int cap=v?std::atoi(v):8192;
    return std::max(chunk,cap);
}


bool prefill_phase_transition(DenseResidency* res,const Geometry& g,const ForwardOptions& opt,
                              bool phase,int nt,cudaStream_t stream,std::string& err){
    if(!res||!res->work||!res->file){err="prefill borrow: no initialized residency";return false;}
    auto& wk=*res->work;auto& sc_pool=wk.sc;auto& gg=res->file->gg;
    const int n_blocks=(int)g.layers.size()-g.n_mtp_blocks;
    const int base_pos=opt.cache?opt.cache->len:0;
    if(wk.slices.pools_full && wk.borrow_phase<0)return true;

        if(!res || !opt.cache || !opt.expert_vram_bytes || opt.poison_scratch) {
            err="prefill borrow requires residency, KV cache, positive expert budget and no poison";return false;
        }
        if(wk.borrow_failed){err="prefill borrow: previous transition failed; restart session";return false;}

        if(wk.borrow_phase!=phase || (phase && nt>wk.borrow_tokens)) {
            const auto began=std::chrono::steady_clock::now();
            if(!wk.slices.drain_window()){err="prefill borrow: verify window drain failed";return false;}
            wk.slices.stop_window();
            if(!wk.slices.drain_async()){err="prefill borrow: async drain failed";return false;}
            wk.slices.stop_async();wk.slices.prefill_ring_close();
            if(cudaDeviceSynchronize()!=cudaSuccess){err="prefill borrow: GPU consumers failed";return false;}
            const auto graph_start=std::chrono::steady_clock::now();
            wk.invalidate_token_graph();for(auto& graph:wk.attn_graph)if(graph){cudaGraphExecDestroy(graph);graph=nullptr;}
            wk.invalidate_window_graphs();
            wk.invalidate_verify_graphs();
            const auto graph_end=std::chrono::steady_clock::now();
            wk.spec_warm_nt=0;
            wk.async_primed=false;
            // The KV/indexer contents and conversation history are deliberately not scratch.
            sc_pool.clear();res->kv_pos_filled=false;
            prefill_moe_release();mat_ctx_release();
            for(auto& b:wk.async_engram_rows)b.reset();for(auto& b:wk.async_engram_valid)b.reset();
            std::fill(wk.async_engram_rows_cap.begin(),wk.async_engram_rows_cap.end(),0);
            std::fill(wk.async_engram_valid_cap.begin(),wk.async_engram_valid_cap.end(),0);
            const auto old_slot_of=wk.slices.slot_of;
            const auto old_pool_of=wk.slices.pool_of;
            std::vector<ExpertSlices::TierPool> old_pools(wk.slices.pools,wk.slices.pools+wk.slices.n_pools);
            std::vector<int32_t> live;
            for(std::size_t key=0;key<wk.slices.slot_of.size();++key)
                if(wk.slices.slot_of[key]>=0)live.push_back((int32_t)key);
            if(phase && wk.borrow_phase!=1){
                wk.decode_slot_of=old_slot_of;wk.decode_pools=old_pools;
                wk.decode_used=wk.slices.used;
                wk.borrowed_observed.assign(old_slot_of.size(),0);
            }
            auto heat=wk.slices.heat;
            const bool had_heat=!wk.slices.slot_of.empty();
            std::vector<ExpertSlices::BlockSizes> sizes;int experts=0;std::size_t smallest=SIZE_MAX;
            for(int il=0;il<n_blocks;++il)if(g.layers[il].expert_ordinal>=0){
                const auto* a=gg.info(blk(il)+"ffn_gate_exps.weight");
                const auto* b=gg.info(blk(il)+"ffn_up_exps.weight");
                const auto* c=gg.info(blk(il)+"ffn_down_exps.weight");
                if(!a||!b||!c){err="prefill borrow: expert metadata missing";return false;}
                experts=(int)a->shape[2];ExpertSlices::BlockSizes bs{il,{}};
                bs.per[0]=tensor_payload_bytes(*a)/experts;bs.per[1]=tensor_payload_bytes(*b)/experts;
                bs.per[2]=tensor_payload_bytes(*c)/experts;sizes.push_back(bs);
                smallest=std::min(smallest,bs.per[0]+bs.per[1]+bs.per[2]);
            }
            std::array<std::size_t,Scratch::kCount> plan{};std::size_t need=0,scratch_need=0,layer_need=0;
            if(phase){
                // MTP prefill needs a hidden row per prompt token, including on a later layer.
                const int head_rows=opt.hidden_all_rows?std::max(nt,kGemmT):kGemmT;
                plan=prefill_workspace_plan(gg,g,nt,opt.cache->max_tokens,base_pos,head_rows,
                    (int)(opt.expert_vram_bytes/smallest),true,kGemmT);
                plan[Scratch::kPos]=std::max(plan[Scratch::kPos],(std::size_t)opt.cache->max_tokens*4);
                for(auto n:plan)scratch_need+=(n+255)&~std::size_t(255);
                for(const auto& b:sizes)layer_need=std::max(layer_need,(b.per[0]+b.per[1]+b.per[2])*experts);
                layer_need=(layer_need+255)&~std::size_t(255);
                need=scratch_need+layer_need;
            }
            std::size_t full_bytes=0;for(const auto& b:sizes)for(auto per:b.per)full_bytes+=per*experts;
            const std::size_t reserve=wk.slices.tier_capacity?wk.slices.tier_capacity:std::min(full_bytes,opt.expert_vram_bytes);
            const char* pct_env=std::getenv("STRATA_HY4_FLASH_PREFILL_LEND_PCT");
            const int pct=pct_env?std::atoi(pct_env):90;
            if(pct<1||pct>100||(phase&&(need>=reserve||need>reserve/100*pct||reserve-need<128*smallest))){
                err="prefill borrow: scratch "+std::to_string(need)+" exceeds loan capacity; lower prefill chunk";return false;
            }
            Buf arena=std::move(wk.slices.tier);
            wk.borrow_failed=true; // a failed refill must never submit logits on the next call
            wk.slices.~ExpertSlices();new (&wk.slices) ExpertSlices();
            wk.slices.open_tier(phase?reserve-need:opt.expert_vram_bytes,n_blocks,experts,g.top_k,sizes,reserve,std::move(arena));
            if(!phase && !wk.decode_pools.empty()){
                if(wk.slices.n_pools!=(int)wk.decode_pools.size()){err="prefill borrow: decode pool count changed";return false;}
                for(int pi=0;pi<wk.slices.n_pools;++pi)if(wk.slices.pools[pi].n!=wk.decode_pools[pi].n){
                    err="prefill borrow: original decode capacity was not restored";return false;
                }
            }
            if(!wk.slices.tier.p){err="prefill borrow: tier allocation failed";return false;}
            if(had_heat)wk.slices.heat=std::move(heat);
            // Kept slot numbers do not change. Move the retained prefixes between planar layouts,
            // using a bounded temporary because cudaMemcpy does not permit overlapping ranges.
            auto& es=wk.slices;std::size_t restored=0,moved=0;int count=0,kept=0;
            const auto relayout_start=std::chrono::steady_clock::now();
            if(!old_pools.empty() && !es.pools_full){
                struct Move {std::size_t from,to,bytes;};std::vector<Move> moves;
                for(int pi=0;pi<es.n_pools;++pi){
                    if(pi>=(int)old_pools.size()){err="prefill borrow: pool topology changed";return false;}
                    const auto& old=old_pools[pi];const auto& now=es.pools[pi];
                    const int rows=std::min(old.n-old.overflow,now.n-now.overflow);
                    for(int r=0;r<3;++r){
                        if(old.stride_r[r]!=now.stride_r[r]){err="prefill borrow: stride changed";return false;}
                        if(rows>0 && old.plane[r]!=now.plane[r])moves.push_back({old.plane[r],now.plane[r],(std::size_t)rows*now.stride_r[r]});
                    }
                }
                const bool grows=!moves.empty() && moves.front().to>moves.front().from;
                if(grows)std::reverse(moves.begin(),moves.end());
                constexpr std::size_t tile=4ull<<20;Buf temp(moves.empty()?0:tile);
                if(!moves.empty() && !temp.p){err="prefill borrow: relayout temporary failed";return false;}
                for(const auto& mv:moves){
                    if((mv.to>mv.from)!=grows){err="prefill borrow: non-monotonic planar move";return false;}
                    for(std::size_t done=0;done<mv.bytes;){
                        const std::size_t bytes=std::min(tile,mv.bytes-done);
                        const std::size_t off=grows?mv.bytes-done-bytes:done;
                        char* base=static_cast<char*>(es.tier.p);
                        if(cudaMemcpyAsync(temp.p,base+mv.from+off,bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess ||
                           cudaMemcpyAsync(base+mv.to+off,temp.p,bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess){
                            err="prefill borrow: relayout copy failed";return false;}
                        done+=bytes;moved+=bytes;
                    }
                }
                if(cudaStreamSynchronize(stream)!=cudaSuccess){err="prefill borrow: relayout completion failed";return false;}
                for(std::size_t key=0;key<old_slot_of.size();++key){
                    const int il=(int)key/experts,slot=old_slot_of[key];if(slot<0 || old_pool_of[il]<0)continue;
                    auto& q=es.pools[es.pool_of[il]];const auto& old=old_pools[old_pool_of[il]];
                    if(slot>=std::min(old.n-old.overflow,q.n-q.overflow))continue;
                    es.slot_of[key]=slot;es.key_of[q.g0+slot]=(int32_t)key;es.used[q.g0+slot]=1;
                    q.next_free=std::max(q.next_free,slot+1);++kept;
                }
            }
            const char* names[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
            const auto refill_start=std::chrono::steady_clock::now();
            if(!es.pools_full){
                if(!es.ensure_copy()){err="prefill borrow: copy stream failed";return false;}

                // Restore the original permanent keys to their original slots. New prompt admissions
                // may use previously empty slots, but cannot displace the saved decode working set.
                const auto& desired=phase?old_slot_of:wk.decode_slot_of;
                for(std::size_t key=0;key<desired.size();++key){
                    const int il=(int)key/experts,e=(int)key%experts;
                    if(desired[key]<0||il<0||il>=n_blocks||es.pool_of[il]<0)continue;
                    auto& q=es.pools[es.pool_of[il]];const int slot=desired[key];
                    const auto& original=phase?old_pools[old_pool_of[il]]:wk.decode_pools[es.pool_of[il]];
                    if(slot>=original.n-original.overflow||slot>=q.n-q.overflow)continue;
                    if(es.slot_of[key]==slot)continue;
                    const int displaced=es.key_of[q.g0+slot];
                    if(displaced>=0)es.slot_of[displaced]=-1;
                    const int prior=es.slot_of[key];
                    if(prior>=0){es.key_of[q.g0+prior]=-1;es.used[q.g0+prior]=0;}
                    for(int r=0;r<3;++r){
                        const auto* ti=gg.info(blk(il)+names[r]);const std::size_t per=tensor_payload_bytes(*ti)/experts;
                        const void* src=gg.at(gg.data_start()+ti->offset+(uint64_t)e*per,per);
                        void* dst=static_cast<char*>(es.tier.p)+q.plane[r]+(std::size_t)slot*q.stride_r[r];
                        if(!src||gg.copy_expert_async(dst,src,per,es.copy)!=cudaSuccess){
                            err="prefill borrow: original expert restore failed";return false;}
                        restored+=per;
                    }
                    es.slot_of[key]=slot;es.key_of[q.g0+slot]=(int32_t)key;es.used[q.g0+slot]=1;
                    q.next_free=std::max(q.next_free,slot+1);++count;
                }
                if(!phase){
                    // Do not return a mostly empty cold decode cache merely because prefill borrowed
                    // its capacity. Admit prompt-observed experts into FREE slots after restoring all
                    // original keys. This uses the same no-eviction policy as prefill_ring_admit.
                    int free_at[ExpertSlices::kMaxPools]={};
                    for(std::size_t key=0;key<wk.borrowed_observed.size();++key){
                        if(!wk.borrowed_observed[key]||es.slot_of[key]>=0)continue;
                        const int il=(int)key/experts,e=(int)key%experts,pi=es.pool_of[il];if(pi<0)continue;
                        auto& q=es.pools[pi];int& slot=free_at[pi];
                        while(slot<q.n-q.overflow && es.key_of[q.g0+slot]>=0)++slot;
                        if(slot>=q.n-q.overflow)continue;
                        for(int r=0;r<3;++r){const auto* ti=gg.info(blk(il)+names[r]);const auto per=tensor_payload_bytes(*ti)/experts;
                            const void* src=gg.at(gg.data_start()+ti->offset+(uint64_t)e*per,per);
                            void* dst=static_cast<char*>(es.tier.p)+q.plane[r]+(std::size_t)slot*q.stride_r[r];
                            if(!src||gg.copy_expert_async(dst,src,per,es.copy)!=cudaSuccess){err="prefill borrow: return admission failed";return false;}
                            restored+=per;
                        }
                        es.slot_of[key]=slot;es.key_of[q.g0+slot]=(int32_t)key;es.used[q.g0+slot]=1;
                        q.next_free=std::max(q.next_free,slot+1);++count;++es.prefill_admitted;++slot;
                    }
                }
                if(cudaStreamSynchronize(es.copy)!=cudaSuccess){err="prefill borrow: restore DMA failed";return false;}
            }
            const auto refill_end=std::chrono::steady_clock::now();

            if(!phase && !wk.decode_pools.empty()){
                // Restoring weights alone is insufficient: resetting these cursors changes the next
                // victims and therefore which experts run on CPU vs GPU. Restore the original policy
                // state while retaining new admissions into slots that were previously empty.
                for(int pi=0;pi<es.n_pools;++pi){
                    auto& q=es.pools[pi];const auto& original=wk.decode_pools[pi];
                    q.hand=original.hand;q.hot_hand=original.hot_hand;q.overflow_at=original.overflow_at;
                }
                for(std::size_t key=0;key<wk.decode_slot_of.size();++key){
                    const int slot=wk.decode_slot_of[key],il=(int)key/experts;
                    if(slot<0 || es.pool_of[il]<0)continue;
                    const auto& q=es.pools[es.pool_of[il]];const auto& original=wk.decode_pools[es.pool_of[il]];
                    if(slot>=original.n-original.overflow)continue;
                    if(es.slot_of[key]!=slot || es.key_of[q.g0+slot]!=(int)key){err="prefill borrow: original permanent slot was not restored";return false;}
                    es.used[q.g0+slot]=wk.decode_used[original.g0+slot];
                }
            }

            const char* verify=std::getenv("STRATA_HY4_FLASH_PREFILL_VERIFY_RESTORE");
            if(verify && std::atoi(verify)!=0){
                std::size_t checked=0;int keys=0;std::vector<unsigned char> image;
                for(std::size_t key=0;key<es.slot_of.size();++key){
                    const int slot=es.slot_of[key],il=(int)key/experts,e=(int)key%experts;
                    if(slot<0||es.pool_of[il]<0)continue;const auto& q=es.pools[es.pool_of[il]];
                    if(slot>=q.n-q.overflow)continue;
                    if(es.key_of[q.g0+slot]!=(int)key){err="prefill borrow: inverse map mismatch";return false;}
                    for(int r=0;r<3;++r){
                        const auto* ti=gg.info(blk(il)+names[r]);const auto per=tensor_payload_bytes(*ti)/experts;
                        const void* expected=gg.at(gg.data_start()+ti->offset+(uint64_t)e*per,per);
                        const void* actual=static_cast<const char*>(es.tier.p)+q.plane[r]+(std::size_t)slot*q.stride_r[r];
                        image.resize(per);
                        if(!expected||cudaMemcpy(image.data(),actual,per,cudaMemcpyDeviceToHost)!=cudaSuccess||std::memcmp(image.data(),expected,per)){
                            err="prefill borrow: expert weight mismatch block="+std::to_string(il)+" expert="+std::to_string(e)+" role="+std::to_string(r);return false;
                        }
                        checked+=per;
                    }++keys;
                }
                std::fprintf(stderr,"HY4_FLASH_PREFILL_RESTORE_VERIFIED phase=%s keys=%d bytes=%zu\n",phase?"prefill":"decode",keys,checked);
            }
            if(!phase){wk.decode_slot_of.clear();wk.decode_pools.clear();wk.decode_used.clear();wk.borrowed_observed.clear();}
            if(phase){
                const std::size_t prefix=es.tier_bytes();
                if(prefix+need>es.tier_capacity){err="prefill borrow: prefix overlaps scratch";return false;}
                sc_pool.lend(static_cast<char*>(es.tier.p)+prefix,scratch_need,plan);
                wk.borrowed_layer=static_cast<char*>(es.tier.p)+prefix+scratch_need;
                wk.borrowed_layer_bytes=layer_need;
            } else {
                wk.borrowed_layer=nullptr;wk.borrowed_layer_bytes=0;
            }
            wk.borrow_phase=phase;wk.borrow_tokens=phase?nt:0;wk.borrow_failed=false;
            if (opt.on_expert_tier) opt.on_expert_tier(es.n_slots, es.tier_bytes());
            std::fprintf(stderr,"HY4_FLASH_PREFILL_LOAN phase=%s tokens=%d arena=%zu expert_bytes=%zu scratch=%zu kept=%d moved_bytes=%zu restored=%d restore_bytes=%zu graph_destroy_ms=%.3f relayout_ms=%.3f refill_ms=%.3f transition_ms=%.3f\n",
                phase?"prefill":"decode",nt,es.tier_capacity,es.tier_bytes(),need,kept,moved,count,restored,
                std::chrono::duration<double,std::milli>(graph_end-graph_start).count(),
                std::chrono::duration<double,std::milli>(refill_start-relayout_start).count(),
                std::chrono::duration<double,std::milli>(refill_end-refill_start).count(),
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count());
        }

    return true;
}

static void open_expert_tier(ExpertSlices& slices, Gguf& gg, const Geometry& g, const ForwardOptions& opt) {
    const int n_blocks = (int)g.layers.size() - g.n_mtp_blocks;
    std::vector<ExpertSlices::BlockSizes> sizes;
    int experts = 0;
    for (int il = 0; il < n_blocks; ++il) {
        if (g.layers[(std::size_t) il].expert_ordinal < 0) continue;
        const TensorInfo* tg = gg.info(blk(il) + "ffn_gate_exps.weight");
        const TensorInfo* tu = gg.info(blk(il) + "ffn_up_exps.weight");
        const TensorInfo* td = gg.info(blk(il) + "ffn_down_exps.weight");
        if (tg == nullptr || tu == nullptr || td == nullptr) continue;
        const uint64_t ne = tg->shape.size() > 2 ? tg->shape[2] : 1;
        if (ne == 0) continue;
        experts = (int) ne;
        const auto per = [ne](const TensorInfo* t) {
            return (std::size_t) (tensor_payload_bytes(*t) / ne);
        };
        // Size each block's pool for its own expert geometry.
        sizes.push_back({il, {per(tg), per(tu), per(td)}});
    }
    // Keyed by (block, expert) over the whole trunk, so a slot taken for block 7's expert 31 is not
    // confused with block 8's.
    slices.open_tier(opt.expert_vram_bytes, n_blocks, experts, g.top_k, sizes);
    if (opt.on_expert_tier && slices.tier.p) opt.on_expert_tier(slices.n_slots, slices.tier_bytes());
}

bool prepare_experts(const std::string& path, const Geometry& g, const ForwardOptions& opt,
                              std::string& err) try {
    if (!opt.dense || opt.dense->file || opt.dense->work) {
        err = "expert initialization requires a fresh residency";
        return false;
    }
    auto& res = *opt.dense;
    res.file = new DenseResidency::Open(path);
    auto& file = res.file->gg;
    if (!file.ok()) {
        err = "cannot open the expert GGUF";
        return false;
    }
    const int blocks = int(g.layers.size()) - g.n_mtp_blocks;
    const char* roles[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    res.work = new DenseResidency::Work();
    auto& s = res.work->slices;
    if (opt.resident_experts) {
        s.exclusive = true;
        open_expert_tier(s, file, g, opt);
        if (!s.tier.p || !s.idx_dev.p || !s.ensure_copy() ||
            (opt.expert_vram_bytes && s.tier_capacity > opt.expert_vram_bytes)) {
            err = "expert VRAM budget cannot hold the routing slots"; return false;
        }
    }
    if (!s.pools_full && !s.configure_cpu_experts()) {
        err = "invalid hy4-flash CPU expert configuration";
        return false;
    }
    if (!opt.resident_experts) {
        // Touch every mapped file page once, without allocating a second copy of the GGUF.
        std::vector<const TensorInfo*> tensors;
        for (int il = 0; il < blocks; ++il) {
            if (g.layers[il].expert_ordinal < 0) continue;
            for (const char* role : roles) {
                const auto* t = file.info(blk(il) + role);
                if (!t) { err = "missing trunk expert tensor: " + blk(il) + role; return false; }
                tensors.push_back(t);
            }
        }
        std::sort(tensors.begin(), tensors.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
        std::vector<uint8_t> scratch;
        for (const auto* t : tensors) {
            const uint64_t from = file.data_start() + t->offset;
            const size_t size = tensor_payload_bytes(*t);
            if (file.mapped()) {
                if (!size || warm_range(file.at(from, size), size) != size) {
                    err = "invalid expert range: " + t->name; return false;
                }
            } else {
                if (scratch.empty()) scratch.resize(1 << 20);
                for (size_t off = 0; off < size;) {
                    const size_t take = std::min(scratch.size(), size - off);
                    if (!file.read_at(from + off, scratch.data(), take)) {
                        err = "cannot preload expert tensor: " + t->name; return false;
                    }
                    off += take;
                }
            }
            res.host_bytes += size;
        }
        std::fprintf(stderr, "HY4_FLASH_EXPERT_HOST mode=full tensors=%zu warmed_bytes=%zu\n",
                     tensors.size(), res.host_bytes);
        return true;
    }

    std::vector<std::string> prefixes(blocks);
    for (int il = 0; il < blocks; ++il) prefixes[il] = blk(il);
    for (int il = 0; il < blocks; ++il) {
        if (g.layers[il].expert_ordinal >= 0 && s.pool_of[il] < 0) {
            err = "expert pool does not cover routing layer " + std::to_string(il);
            return false;
        }
    }
    if (s.pools_full) {
        if (!s.prefill_all(file, prefixes)) { err = "full expert tier upload failed"; return false; }
        std::fprintf(stderr, "HY4_FLASH_EXPERT_HOST mode=resident pool_bytes=0 spare_bytes=0\n");
        std::fprintf(stderr, "HY4_FLASH_RESIDENT gpu_slots=%d ram_bytes=0 file_experts=0\n", s.n_slots);
        return true;
    }

    // Distribute each pool's permanent slots across its layers before choosing the RAM complement.
    std::vector<int> gpu_count(blocks);
    s.ram_offset.assign(s.slot_of.size(), core::detail::kNoCacheComplement);
    const char* direct_env = std::getenv("STRATA_HY4_FLASH_RESIDENT_DIRECT_DMA");
    size_t spare_bytes = 0;
    if (!direct_env || std::atoi(direct_env) != 0) {
        for (int pi = 0; pi < s.n_pools; ++pi) spare_bytes += s.pools[pi].slot_bytes;
    }
    size_t ram_bytes = 0;
    int ram_count = 0, gpu_count_total = 0;
    for (int e = 0; e < s.n_expert; ++e) {
        for (int il = 0; il < blocks; ++il) {
            if (s.pool_of[il] < 0) continue;
            auto& q = s.pools[s.pool_of[il]];
            if (q.next_free < q.n - q.overflow) {
                ++q.next_free;
                ++gpu_count[il];
                ++gpu_count_total;
            } else {
                s.ram_offset[il*s.n_expert + e] = ram_bytes;
                ram_bytes += q.slot_bytes;
                ++ram_count;
            }
        }
    }
    if (ram_bytes) {
        HostMemory memory;
        const size_t staging_bytes = 2 * s.slot_bytes;
        const size_t required = ram_bytes + spare_bytes + staging_bytes;
        if (host_available_memory(memory) && required > memory.available) {
            err = "resident experts require " + std::to_string(required) +
                  " bytes of host RAM including swap and staging slots, " +
                  std::to_string(memory.available) + " bytes available; increase --expert-vram or provide more RAM";
            return false;
        }
        if (spare_bytes) {
            size_t total = ram_bytes;
            for (int i = 0; i < s.n_pools; ++i) {
                s.pools[i].ram_spare = total;
                total += s.pools[i].slot_bytes;
            }
            const auto status = cudaHostAlloc((void**)&s.pinned_ram, total, cudaHostAllocDefault);
            if (status == cudaSuccess) {
                const bool overlap = [] {
                    if (const char* value = std::getenv("STRATA_HY4_FLASH_RESIDENT_PLANE_STREAMS"))
                        return std::atoi(value) != 0;
                    int device = 0;
                    cudaDeviceProp props{};
                    if (cudaGetDevice(&device) != cudaSuccess ||
                        cudaGetDeviceProperties(&props, device) != cudaSuccess)
                        return false;
                    constexpr size_t gib = size_t(1) << 30;
                    return std::strstr(props.name, "RTX 4090") != nullptr &&
                           props.totalGlobalMem >= 45 * gib && props.totalGlobalMem <= 50 * gib;
                }();
                if (overlap) {
                    for (auto& stream : s.plane_streams)
                        if (cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess) {
                            err="cannot create expert exchange stream";return false;
                        }
                    if (cudaEventCreateWithFlags(&s.planes_ready,cudaEventDisableTiming)!=cudaSuccess) {
                        err="cannot create expert exchange event";return false;
                    }
                    std::fprintf(stderr,"HY4_FLASH_RESIDENT exchange_plane_streams=3\n");
                }
                std::fprintf(stderr, "HY4_FLASH_RESIDENT transport=direct pinned_bytes=%zu spare_bytes=%zu\n",
                             total, total - ram_bytes);
            } else {
                std::fprintf(stderr, "HY4_FLASH_RESIDENT host pin failed: %s; using pageable pool\n",
                             cudaGetErrorString(status));
                cudaGetLastError();
            }
        }
        if (!s.pinned_ram) {
            s.ram_pool.reset(new uint8_t[ram_bytes]);
            std::fprintf(stderr, "HY4_FLASH_RESIDENT transport=staged pageable_bytes=%zu\n", ram_bytes);
        }
    }
    if (cudaHostAlloc((void**)&s.exchange, 2*s.slot_bytes, cudaHostAllocDefault) != cudaSuccess) {
        err = "expert exchange staging allocation failed"; return false;
    }
    size_t uploads = 0;
    std::vector<int> next(s.n_pools);
    for (int il = 0; il < blocks; ++il) {
        const int pi = s.pool_of[il];
        if (pi < 0) continue;
        auto& q = s.pools[pi];
        for (int e = 0; e < gpu_count[il]; ++e) {
            const int key = il*s.n_expert + e, slot = next[pi]++;
            s.slot_of[key] = slot;
            s.key_of[q.g0+slot] = key;
        }
        for (int r = 0; r < 3; ++r) {
            const auto* tensor = file.info(prefixes[il] + roles[r]);
            const size_t per = tensor_payload_bytes(*tensor) / s.n_expert;
            for (int e = 0; e < s.n_expert; ++e) {
                const uint64_t from = file.data_start() + tensor->offset + uint64_t(e)*per;
                const int key = il*s.n_expert + e, slot = s.slot_of[key];
                if (slot >= 0) {
                    auto* dst = static_cast<uint8_t*>(s.tier.p) + q.plane[r] + size_t(slot)*q.stride_r[r];
                    // Read the next plane or RAM expert while the previous GPU upload is in flight.
                    const size_t bank = uploads++ % 2;
                    auto* buffer = s.exchange + bank * s.slot_bytes;
                    if ((bank == 0 && cudaStreamSynchronize(s.copy) != cudaSuccess) ||
                        !file.pread_at(from, buffer, per) ||
                        cudaMemcpyAsync(dst, buffer, per, cudaMemcpyHostToDevice, s.copy) != cudaSuccess) {
                        err = "GPU expert initialization failed"; return false;
                    }
                } else if (auto* plane = s.ram_plane(key, r)) {
                    if (!file.pread_at(from, plane, per)) {
                        err = "RAM expert initialization failed"; return false;
                    }
                }
            }
        }
    }
    if (cudaStreamSynchronize(s.copy) != cudaSuccess) {
        err = "initial expert upload failed";
        return false;
    }
    std::fprintf(stderr, "HY4_FLASH_EXPERT_HOST mode=resident pool_bytes=%zu spare_bytes=%zu\n",
                 ram_bytes + (s.pinned_ram ? spare_bytes : 0),
                 s.pinned_ram ? spare_bytes : 0);
    std::fprintf(stderr, "HY4_FLASH_RESIDENT gpu_slots=%d gpu_bytes=%zu ram_experts=%d ram_bytes=%zu "
                         "file_experts=0 exchange_bytes=%zu\n", gpu_count_total, s.tier_capacity,
                         ram_count, ram_bytes, 2*s.slot_bytes);
    return true;
} catch (const std::exception& e) {
    err = e.what(); return false;
}

static bool forward_prefill_impl(const std::string& gguf_path, const Geometry& g, const std::string& pack_dir,
                     const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err,
                     const ForwardOptions& opt) try {
    // A capture is only ever valid after a pass that finished; every early return below leaves it false.
    if (opt.dfly != nullptr) opt.dfly->valid = false;
    const bool allow_borrow = prefill_borrow_on() && !opt.resident_experts;
    if(!opt.exact_columns && layer_major_on() && (!auto_chunk_on() || g_auto_dispatch) && !g_layer_major && (tokens.size()>(std::size_t)prefill_compute_chunk() || (allow_borrow && tokens.size()>(std::size_t)kGemmT))){
        if(!opt.cache || !opt.dense || (opt.want_logits && !opt.logits_last_only) || opt.on_block || opt.teacher_state || opt.on_routing || opt.poison_scratch){
            err="layer-major requires cached last-row inference without diagnostic hooks";return false;
        }
        const int nb=(int)g.layers.size()-g.n_mtp_blocks,chunk=prefill_compute_chunk();
        const int original=opt.cache->len;const std::size_t width=(std::size_t)g.hidden_dim*g.hc_mult;
        if(original<0 || (original>0 && (!opt.history || opt.history->size()<(std::size_t)original))){
            err="layer-major: the cached prefix needs its token history";return false;}
        if((std::size_t)original+tokens.size()>(std::size_t)opt.cache->max_tokens){
            err="layer-major: prompt exceeds the reserved KV context";return false;}
        if(opt.dense->work && opt.dense->work->idx && !indexer_cache_rewind(*opt.dense->work->idx,original)){
            err="layer-major: indexer cannot rewind to the requested prefix";return false;}
        std::vector<int32_t> history;
        if(original>0)history.assign(opt.history->begin(),opt.history->begin()+original);
        history.insert(history.end(),tokens.begin(),tokens.end());
        LayerMajorCall lm;float* host=nullptr;
        lm.state_device=layer_major_state_device();
        lm.state_bytes=width*tokens.size()*sizeof(float);
        if(!lm.state_device && cudaHostAlloc((void**)&host,lm.state_bytes,cudaHostAllocDefault)!=cudaSuccess){
            err="layer-major: host activation buffer allocation failed";return false;}
        struct Scope{float* host;~Scope(){g_layer_major=nullptr;if(host)cudaFreeHost(host);}}scope{host};
        g_layer_major=&lm;logits_out.clear();
        const auto began=std::chrono::steady_clock::now();
        for(int il=0;il<nb;++il){
            lm.layer=il;
            for(std::size_t at=0;at<tokens.size();at+=(std::size_t)chunk){
                const std::size_t n=std::min((std::size_t)chunk,tokens.size()-at);
                std::vector<int32_t> part(tokens.begin()+at,tokens.begin()+at+n);
                ForwardOptions co=opt;
                co.history=&history;
                // Keep the baseline head's multi-column arithmetic for a <=8-token tail, then
                // return only its last row to the superchunk caller.
                co.logits_last_only=n>(std::size_t)kGemmT;
                const bool final_layer = il == nb - 1;
                const bool final_chunk = at + n == tokens.size();
                // Selections belong to individual sparse layers. Collect the last row at each
                // of them, then publish the hidden state and complete the capture at the head.
                co.dfly = final_chunk ? opt.dfly : nullptr;
                co.want_logits = opt.want_logits && final_layer && final_chunk;
                co.hidden_out = nullptr;
                Buf tail_hidden;
                if (final_layer && opt.hidden_out) {
                    if (opt.hidden_all_rows || !opt.logits_last_only) {
                        co.hidden_all_rows = true;
                        co.hidden_out = opt.hidden_out + at * g.hidden_dim;
                    } else if (final_chunk) {
                        // The short-tail head computes all rows. Keep its arithmetic but
                        // copy only the last row to a caller with space for one.
                        if (!co.logits_last_only && n > 1) {
                            tail_hidden = Buf(n * g.hidden_dim * sizeof(float));
                            if (!tail_hidden.p) {
                                opt.cache->len = original;
                                err = "layer-major: short-tail hidden buffer allocation failed";
                                return false;
                            }
                        }
                        co.hidden_out = tail_hidden.p ? tail_hidden.as<float>() : opt.hidden_out;
                    }
                }
                lm.state_offset=at*width;lm.state=host?host+lm.state_offset:nullptr;opt.cache->len=original+(int)at;
                std::vector<float> row;
                if(!forward_prefill(gguf_path,g,pack_dir,part,row,err,co)){
                    opt.cache->len=original;logits_out.clear();return false;}
                const size_t last_row = co.logits_last_only ? 0 : n - 1;
                if (tail_hidden.p && cudaMemcpy(opt.hidden_out, co.hidden_out + last_row * g.hidden_dim,
                                                g.hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    opt.cache->len = original;
                    err = "layer-major: last hidden row copy failed";
                    return false;
                }
                if (co.want_logits) {
                    if (opt.device_logits_out)
                        *opt.device_logits_out += last_row * g.vocab_size;
                    else
                        logits_out.assign(row.end() - g.vocab_size, row.end());
                }
            }
        }
        opt.cache->len=original+(int)tokens.size();
        std::fprintf(stderr,"HY4_FLASH_LAYER_MAJOR tokens=%zu chunk=%d layers=%d expert_h2d=%zu expert_d2d=%zu host_state=%zu device_state=%zu total_ms=%.3f streamed_experts=%d\n",
            tokens.size(),chunk,nb,lm.h2d,lm.d2d,lm.state_device?0:lm.state_bytes,lm.state_device?lm.state_bytes:0,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count(), lm.stream_experts);
        return true;
    }
    g_poison = opt.poison_scratch;
    // Measure host-side work with wall-clock time.
    PhaseTime* pt = opt.timing;
    const auto t_call = std::chrono::steady_clock::now();
    auto tick = [](double& into, const std::chrono::steady_clock::time_point& from) {
        into += std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
    };
    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        err = "forward_prefill: no CUDA stream";
        return false;
    }
    struct StreamGuard {
        cudaStream_t s;
        DenseResidency* packed_owner = nullptr;
        int exceptions = std::uncaught_exceptions();
        bool verify_barrier = false, completed = false;
        ~StreamGuard() {
            // Only the opt-in packed path arms this. Reuse the existing async abort protocol:
            // publish error first so already-enqueued GPU waiters can retire before any sync.
            if (packed_owner && std::uncaught_exceptions() > exceptions && packed_owner->work) {
                auto& wk = *packed_owner->work;
                wk.slices.async_fail(32, "dense arena forward exception");
                wk.slices.window_fail("forward exception");
                cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
                if (cudaStreamIsCapturing(s, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
                    cudaGraph_t unfinished = nullptr;
                    (void)cudaStreamEndCapture(s, &unfinished);
                    if (unfinished) (void)cudaGraphDestroy(unfinished);
                }
                wk.slices.stop_window();wk.slices.stop_async();
            }
            // An exception can interrupt the shared-expert branch before its join. A failed verify
            // must drain that stream too, before the caller resets or releases captured buffers.
            if (verify_barrier && !completed) cudaDeviceSynchronize();
            cudaStreamDestroy(s);
        }
    } guard{stream};
    guard.verify_barrier = opt.exact_columns || opt.dfly != nullptr;

    // Enable programmatic dependent launch only on supported devices.
    struct PdlScope {
        bool prev;
        explicit PdlScope(bool on) : prev(kernels::pdl_scope()) { kernels::pdl_scope() = on; }
        ~PdlScope() { kernels::pdl_scope() = prev; }
    } pdl_pass(kernels::pdl_supported());

    const int n_embd = g.hidden_dim, hc = g.hc_mult, vocab = g.vocab_size;
    const int n_blocks = (int) g.layers.size() - g.n_mtp_blocks;   // the trunk; the draft block is separate

    // The dense weights, kept across calls when the caller asked for it - and with them the open file and
    // its mapping, which is the part a decode step cannot do without. One slot per trunk block and one more
    // for the embedding and head; `take` below fills a slot the first time it is asked for a name.
    DenseResidency* res = opt.dense;
    if (res != nullptr && (int) res->slots.size() != n_blocks + 1) {
        for (DenseResidency::Slot* s : res->slots) delete s;
        res->slots.clear();
        res->bytes = 0;
        for (int i = 0; i <= n_blocks; ++i) res->slots.push_back(new DenseResidency::Slot());
        if (res->work != nullptr) res->work->attn_graph.assign((std::size_t) n_blocks, nullptr);
    }
    std::unique_ptr<Gguf> own_gg;
    std::unique_ptr<DenseResidency::Work> own_work;
    if (res != nullptr) {
        if (res->file == nullptr) res->file = new DenseResidency::Open(gguf_path);
        if (res->work == nullptr) res->work = new DenseResidency::Work();
        if ((int) res->work->attn_graph.size() != n_blocks)
            res->work->attn_graph.assign((std::size_t) n_blocks, nullptr);
        // Invalidate captured graphs before multi-token scratch growth can change device addresses.
        if ((int) tokens.size() != res->work->captured_nt()) {
            res->work->invalidate_token_graph();
            for (cudaGraphExec_t& g : res->work->attn_graph)
                if (g != nullptr) {
                    cudaGraphExecDestroy(g);
                    g = nullptr;
                }
        }
    } else {
        own_gg.reset(new Gguf(gguf_path));
        own_work.reset(new DenseResidency::Work());
    }
    Gguf& gg = res != nullptr ? res->file->gg : *own_gg;
    DenseResidency::Work& wk = res != nullptr ? *res->work : *own_work;
    if(!wk.slices.drain_window()){err="window service drain failed";return false;}
    // Prompt reuse does not itself invalidate window captures. Address generation,
    // KV identity, Engram buffer growth and indexer-mode changes do so below.
    Scratch& sc_pool = wk.sc;
    if (!gg.ok()) {
        err = "forward_prefill: cannot open " + gguf_path;
        return false;
    }
    const int nt = (int) tokens.size();
    if (nt <= 0) {
        err = "forward_prefill: no tokens";
        return false;
    }

    // The queries' absolute positions. For a prefill that is 0..nt-1; for decode it continues from the
    // cache, so RoPE and the causal mask see the real position instead of 0 again.
    const int base_pos = opt.cache != nullptr ? opt.cache->len : opt.first_pos;
    static const char* async_env=std::getenv("STRATA_HY4_FLASH_ASYNC_MOE");
    const bool async_option=async_env && std::atoi(async_env)!=0;
    static const bool dfly_graph_option = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_GRAPH");
        if (v && std::atoi(v) == 0) return false;
        // These diagnostics read back or deliberately use a host-length attention loop.
        for (const char* name : {"STRATA_HY4_FLASH_DFLY_PQ_LOOP", "STRATA_HY4_FLASH_DFLY_PQ_CHECK",
                                 "STRATA_HY4_FLASH_DFLY_MOE_CHECK", "STRATA_HY4_FLASH_FUSED_CHECK",
                                 "STRATA_HY4_FLASH_SPLIT_CHECK", "STRATA_HY4_FLASH_SEL_DIAG"}) {
            const char* d = std::getenv(name);
            if (d && std::atoi(d) != 0) return false;
        }
        if (std::getenv("STRATA_HY4_FLASH_DIAG_FA_LIBRARY") || std::getenv("STRATA_DIAG_ROUTER_LIBRARY"))
            return false;
        return true;
    }();
    const bool verify_prepare = dfly_graph_option && graphs_enabled() && opt.exact_columns &&
        opt.dfly != nullptr && opt.dfly->mode == DflyCapture::Mode::verify_window && nt >= 1 && nt <= DflyDrafter::kMaxK + 1 &&
        res != nullptr && opt.cache != nullptr && kv_format(*opt.cache) != kKvF32 && !g_layer_major &&
        !opt.on_block && !opt.on_attn && !opt.on_routing && !opt.teacher_state && !pt && !opt.poison_scratch;
    if (base_pos == 0) wk.invalidate_verify_graphs();
    static const int cpu_requested=[] {
        const char* e=std::getenv("STRATA_HY4_FLASH_CPU_MISS_MAX");return e?std::atoi(e):0;
    }();
    if(cpu_requested<0 || cpu_requested>16 || (cpu_requested && !async_option)) {
        err="STRATA_HY4_FLASH_CPU_MISS_MAX must be 0..16 and requires STRATA_HY4_FLASH_ASYNC_MOE=1";
        return false;
    }
    // Single-phase publication requires the device expert plan.
    static const bool single_phase = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ASYNC_SINGLE_PHASE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    static const bool device_plan_on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ASYNC_DEVICE_PLAN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (single_phase && (!async_option || !device_plan_on)) {
        err = "forward_prefill: STRATA_HY4_FLASH_ASYNC_SINGLE_PHASE needs STRATA_HY4_FLASH_ASYNC_MOE=1 and "
              "STRATA_HY4_FLASH_ASYNC_DEVICE_PLAN=1 - without the device plan the wait-all route cannot "
              "distinguish an all-hit epoch from a real miss";
        return false;
    }
    // The mode is baked into captured graphs. Runtime changes require a clean request boundary.
    if (wk.single_phase != single_phase) {
        if (!wk.slices.drain_async()) { err = "single-phase transition drain failed"; return false; }
        for (auto& graph : wk.attn_graph) if (graph) { cudaGraphExecDestroy(graph); graph = nullptr; }
        wk.invalidate_token_graph();
        wk.async_primed = false; wk.single_phase = single_phase;
        std::fprintf(stderr, "ASYNC_SINGLE_PHASE active=%d graphs_invalidated=1\n", (int) single_phase);
    }
    // Guard the combination of asynchronous expert scheduling and PDL.
    static const bool pdl_on = [] {
        const char* v = std::getenv("STRATA_DF_PDL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    static const bool async_allow_pdl = [] {
        const char* v = std::getenv("HY4_FLASH_ASYNC_ALLOW_PDL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (async_option && pdl_on && !async_allow_pdl) {
        err = "forward_prefill: STRATA_HY4_FLASH_ASYNC_MOE and STRATA_DF_PDL are not validated together - "
              "set HY4_FLASH_ASYNC_ALLOW_PDL=1 to run them anyway, or unset one of them";
        return false;
    }
    // The async scheduler's priming is per T1 step and is reset by anything wider, UNCHANGED: a prefill
    // chunk must still drain and re-prime whether or not it also drops a graph.
    if(nt!=1 || base_pos==0) {
        wk.invalidate_token_graph();
        wk.async_primed=false;
        if(!wk.slices.drain_async()){err="async MoE prefill drain failed";return false;}
    }
    // The captured set, on the other hand, only has to go when the WIDTH it was built for changes (or the
    // sequence restarts). While nothing is captured `captured_nt()` is 1, so for a T1-only capture policy
    // this fires on exactly the calls the old `nt!=1` test fired on.
    if(nt!=wk.captured_nt() || base_pos==0) {
        for(auto& graph:wk.attn_graph)if(graph){cudaGraphExecDestroy(graph);graph=nullptr;}
        wk.graph_nt=nt;
    }


    std::vector<int32_t> pos((std::size_t) nt);
    for (int i = 0; i < nt; ++i) pos[(std::size_t) i] = base_pos + i;
    S d_pos = sc_pool.get(Scratch::kPos, (std::size_t) nt * 4);
    cudaMemcpy(d_pos.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);

    // Cached positions are absolute. Initialize the position table once to capacity.
    const int n_kv = opt.cache != nullptr ? opt.cache->len + nt : nt;
    const int kv_stride = opt.cache != nullptr ? opt.cache->max_tokens : n_kv;
    S d_kv_pos, d_len;
    if (opt.cache != nullptr) {
        if (opt.cache->len + nt > opt.cache->max_tokens) {
            err = "forward_prefill: the cache holds " + std::to_string(opt.cache->max_tokens) +
                  " positions and this call needs " + std::to_string(opt.cache->len + nt);
            return false;
        }
        d_kv_pos = sc_pool.get(Scratch::kKvPos, (std::size_t) kv_stride * 4);
        if (!res->kv_pos_filled) {
            std::vector<int32_t> kvp((std::size_t) kv_stride);
            for (int i = 0; i < kv_stride; ++i) kvp[(std::size_t) i] = i;
            cudaMemcpy(d_kv_pos.p, kvp.data(), kvp.size() * 4, cudaMemcpyHostToDevice);
            res->kv_pos_filled = true;
        }
        // The two counts the attention half needs that change every token: where to append, and how many
        // keys there now are. On the device so a captured graph can read them instead of baking them.
        const int32_t meta[2] = {(int32_t) opt.cache->len, (int32_t) (opt.cache->len + nt)};
        d_len = sc_pool.get(Scratch::kLen, 2 * sizeof(int32_t));
        cudaMemcpy(d_len.p, meta, sizeof meta, cudaMemcpyHostToDevice);
    }

    const float rms_eps = g.rms_norm_eps, hc_eps = g.hc_eps, magnitude = g.hc_magnitude;

    DenseResidency::Slot* head_slot = res != nullptr ? res->slots[(std::size_t) n_blocks] : nullptr;
    std::size_t* budget = res != nullptr ? &res->bytes : nullptr;

    // ---- the VRAM tier, opened once per residency, sized from the file's own strides.
    if (wk.slices.slot_bytes == 0) open_expert_tier(wk.slices, gg, g, opt);

    // Use one timestamp across all layers so layer-major prefill does not age early layers faster.
    if (wk.slices.hot_cache && !(auto_chunk_on() && !g_auto_dispatch && nt>kGemmT) && (!g_layer_major || g_layer_major->layer==0)) wk.slices.heat.advance_tokens((uint64_t)nt);

    // Preload all trunk experts when the tier can hold them.
    if (wk.slices.pools_full && !wk.slices.complete) {
        std::vector<std::string> prefixes((std::size_t) n_blocks);
        for (int il = 0; il < n_blocks; ++il) prefixes[(std::size_t) il] = blk(il);
        wk.slices.prefill_all(gg, prefixes);
    }

    static const bool pin_expert_mmap = [] {
        const char* p = std::getenv("STRATA_HY4_FLASH_PIN_EXPERT_MMAP");
        return p && std::atoi(p) != 0;
    }();
    static const bool prefill_direct_dma = [] {
        const char* v=std::getenv("STRATA_HY4_FLASH_PREFILL_DIRECT_DMA");
        return v && std::atoi(v)!=0;
    }();
    if (!opt.resident_experts && pin_expert_mmap && (async_option || (nt>1 && prefill_direct_dma)) &&
        !wk.slices.pools_full && !gg.expert_pin_attempted()) {
        std::vector<std::pair<uint64_t, std::size_t>> tensors;
        const char* roles[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (int il = 0; il < n_blocks; ++il) {
            if (g.layers[(std::size_t) il].expert_ordinal < 0) continue;
            for (const char* role : roles) {
                const TensorInfo* t = gg.info(blk(il) + role);
                if (!t) { err = "pin expert mmap: missing expert tensor"; return false; }
                tensors.emplace_back(gg.data_start() + t->offset, (std::size_t) tensor_payload_bytes(*t));
            }
        }
        // After the ordinary file warmup, but before the initial prompt computation. Failures
        // leave no partial registration and preserve staged supply; startup costs remain visible.
        if (!gg.pin_expert_ranges(tensors))
            std::fprintf(stderr, "HY4_FLASH_PIN_EXPERT_MMAP initialization unavailable; using staged copies\n");
    }

    static const bool dense_arena_on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_DENSE_ARENA");
        return v && std::atoi(v) != 0;
    }();
    if (dense_arena_on) {
        guard.packed_owner = res;
        if (gg.dense_arena_failed()) { err="dense arena is failed; recreate the engine"; return false; }
    }
    if (dense_arena_on && !gg.dense_arena_ready()) {
        if (!res || res->bytes != 0 || wk.warm) {
            err = "dense arena requires a fresh persistent DenseResidency"; return false;
        }
        static const char* const suffixes[] = {
            "attn_gate.weight",
            "attn_k_b.weight",
            "attn_kv_a_mqa.weight",
            "attn_kv_a_norm.weight",
            "attn_norm.weight",
            "attn_output.weight",
            "attn_q_a.weight",
            "attn_q_a_norm.weight",
            "attn_q_b.weight",
            "attn_sinks.weight",
            "attn_v_b.weight",
            "engram_gate.weight",
            "engram_hc_scale.weight",
            "engram_kv.weight",
            "engram_kv_b.weight",
            "engram_lookup_norm.weight",
            "engram_pre_gamma.weight",
            "exp_probs_b.bias",
            "ffn_down.weight",
            "ffn_down_shexp.weight",
            "ffn_gate.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_norm.weight",
            "ffn_routed_down.weight",
            "ffn_routed_up.weight",
            "ffn_up.weight",
            "ffn_up_shexp.weight",
            "hc_attn_base.weight",
            "hc_attn_fn.weight",
            "hc_attn_scale.weight",
            "hc_ffn_base.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_scale.weight",
            // Enumerate indexer weights only for sparse attention blocks.
            "indexer.attn_k.weight",
            "indexer.attn_q_b.weight",
            "indexer.k_norm.weight",
            "indexer.proj.weight",
            "indexer_compressor_ape.weight",
            "indexer_compressor_gate.weight",
            "indexer_compressor_norm.weight",
            "indexer_compressor_value.weight",
            "post_attention_norm.weight",
            "post_ffw_norm.weight"
        };
        std::vector<std::string> names = {"output.weight", "output_hc_base.weight", "output_hc_fn.weight", "output_hc_scale.weight", "output_norm.weight", "token_embd.weight"};
        for (int il = 0; il < n_blocks; ++il)
            for (const char* suffix : suffixes) names.push_back(blk(il) + suffix);
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        if (!gg.prepare_dense_arena(names, err)) return false;
    }


    // Resolve automatic chunking AFTER the real dense/expert allocations, before mutating KV or
    // indexer contents. Recursive calls perform the original numerical path at the chosen extent.
    if(auto_chunk_on() && !g_auto_dispatch && nt>kGemmT) {
        if(!res || !opt.cache || !dense_arena_on || opt.on_block || opt.teacher_state ||
           opt.on_routing || opt.poison_scratch || (opt.want_logits && !opt.logits_last_only)) {
            err="automatic prefill chunk requires dense arena, cached last-row inference and no hooks";
            return false;
        }
        std::size_t free_bytes=0,total_bytes=0;
        if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess){err="automatic chunk: memory query failed";return false;}
        std::int64_t heads=0;int slots=0;std::size_t layer_bytes=0;
        for(int pi=0;pi<wk.slices.n_pools;++pi)slots=std::max(slots,wk.slices.pools[pi].n);
        for(int il=0;il<n_blocks;++il){
            heads=std::max<std::int64_t>(heads,g.layers[il].n_heads);
            if(g.layers[il].expert_ordinal>=0){
                std::size_t bytes=0;
                for(const char* role:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}){
                    const auto* ti=gg.info(blk(il)+role);
                    if(!ti){err="automatic chunk: missing expert tensor";return false;}
                    bytes+=tensor_payload_bytes(*ti);
                }
                layer_bytes=std::max(layer_bytes,bytes);
            }
        }
        const std::int64_t widest=std::max<std::int64_t>({(std::int64_t)n_embd*hc,g.q_lora_rank,g.v_head_dim*heads,g.dense_ff});
        const std::int64_t welems=std::max<std::int64_t>((std::int64_t)g.dense_ff*n_embd,(std::int64_t)g.q_lora_rank*g.qk_head_dim*heads);

        std::size_t lendable=0;
        if(allow_borrow && !wk.slices.pools_full){
            if(!layer_major_on()){err="prefill borrow currently requires layer-major";return false;}
            const char* pe=std::getenv("STRATA_HY4_FLASH_PREFILL_LEND_PCT");const int pct=pe?std::atoi(pe):90;
            if(pct<1||pct>100){err="prefill lend percent must be 1..100";return false;}
            std::size_t smallest=SIZE_MAX;
            for(int pi=0;pi<wk.slices.n_pools;++pi){const auto& q=wk.slices.pools[pi];smallest=std::min(smallest,q.stride_r[0]+q.stride_r[1]+q.stride_r[2]);}
            const auto arena=wk.slices.tier_capacity;
            if(arena>128*smallest)lendable=std::min(arena/100*pct,arena-128*smallest);
        }
        int chosen=0;std::size_t selected_need=0;
        // A loan spans all outer chunks of one prompt. Keep its already-provisioned compute
        // extent until logits return the loan; replanning against its own live backend would
        // double-charge startup headroom and change numerical shapes halfway through a prompt.
        if(wk.auto_prompt_chunk>0)chosen=wk.auto_prompt_chunk;
        else if(allow_borrow && wk.borrow_phase==1)
            chosen=std::min(auto_chunk_cap(),wk.borrow_tokens);
        for(int candidate:{8192,4096,2048,1024,512,256,128}){
            if(chosen)break;
            if(candidate>auto_chunk_cap())continue;
            const int ct=std::min(candidate,nt);
            const bool relayout=allow_borrow && !wk.slices.pools_full &&
                (wk.borrow_phase!=1 || ct>wk.borrow_tokens);
            std::size_t available=free_bytes;
            const auto owned=mat_ctx_owned_bytes();
            if(relayout){
                // These allocations are synchronously released before the new phase allocates.
                // Credit only owned buffers; borrowed views are already inside the expert arena.
                available+=owned;
                for(int i=0;i<Scratch::kCount;++i)if(sc_pool.buf[i].owns)available+=sc_pool.cap[i];
                for(auto n:wk.async_engram_rows_cap)available+=n;
                for(auto n:wk.async_engram_valid_cap)available+=n;
            }
            const int logit_rows = relayout ? kGemmT : opt.want_logits ? 1 : 0;
            const int min_head_rows = relayout ? kGemmT : 1;
            const int head_rows = opt.hidden_all_rows || (opt.hidden_out && !opt.logits_last_only)
                ? std::max(ct,min_head_rows) : min_head_rows;
            const auto plan=prefill_workspace_plan(gg,g,ct,kv_stride,base_pos,head_rows,slots,
                                                  layer_major_on() || wk.slices.complete,logit_rows);
            // Scratch growth constructs the replacement before freeing the old slot. Account for
            // that transient as well as final growth, rather than silently exceeding the budget.
            std::size_t need=0,overlap=0;
            for(int i=0;i<Scratch::kCount;++i)if(plan[i]>sc_pool.cap[i]){
                need+=plan[i]-sc_pool.cap[i];overlap=std::max(overlap,sc_pool.cap[i]);
            }
            need+=overlap;
            if(allow_borrow && !wk.slices.pools_full){
                // The arena already includes the currently borrowed scratch. Count its complete
                // replacement, not just growth, before crediting the lending capacity once.
                need=0;for(auto bytes:plan)need+=(bytes+255)&~std::size_t(255);
            }
            const auto mat=mat_ctx_plan_bytes(ct,widest,welems);
            if(relayout)need+=mat;else if(mat>owned)need+=mat-owned;
            if(!wk.slices.complete)need+=layer_major_on()?layer_bytes:layer_bytes/8; // whole layer or <=64-expert ring
            if(!wk.idx && g.idx_kpool>0){
                int sparse=0;for(const auto& l:g.layers)if(l.attention!=Attention::Sliding)++sparse;
                need+=(std::size_t)sparse*g.idx_key_dim*4*((kv_stride+g.idx_kpool-1)/g.idx_kpool+1+2*g.idx_kpool);
            }
            if(async_option)for(int il=0;il<n_blocks;++il)if(g.layers[il].engram_table>=0){
                const std::size_t rows=(std::size_t)g.engram_heads*g.engram_head_dim*ct*4,valid=(std::size_t)g.engram_heads*ct*4;
                const auto rc=il<(int)wk.async_engram_rows_cap.size()?wk.async_engram_rows_cap[il]:0;
                const auto vc=il<(int)wk.async_engram_valid_cap.size()?wk.async_engram_valid_cap[il]:0;
                if(relayout || rows>rc)need+=rows; if(relayout || valid>vc)need+=valid;
            }
            // The super-chunk residual is allocated separately after entering the loan. It must
            // fit OUTSIDE the expert arena; do not count it as borrowed scratch or reclaimable state.
            if(layer_major_on() && layer_major_state_device() &&
               (nt>candidate || allow_borrow))
                need+=(std::size_t)n_embd*hc*nt*sizeof(float);
            // cuBLAS/MMQ internal allocations, small maps/events and allocator rounding.
            need+=512ull<<20;

            std::size_t scratch_bytes=0;for(auto bytes:plan)scratch_bytes+=(bytes+255)&~std::size_t(255);
            const auto loan_bytes=scratch_bytes+layer_bytes;
            const bool fits_loan=!allow_borrow || wk.slices.pools_full ||
                (loan_bytes<=lendable && need>=loan_bytes && need-loan_bytes<=available);
            if(need<=available+lendable && fits_loan){
                chosen=candidate;selected_need=need;break;
            }
        }
        if(!chosen){err="automatic chunk: even 128 tokens do not fit beside the expert tier; lower expert budget";return false;}
        std::fprintf(stderr,"HY4_FLASH_PREFILL_AUTO tokens=%d chunk=%d free=%zu planned_extra=%zu lendable=%zu layer_major=%d\n",
                     nt,chosen,free_bytes,selected_need,lendable,(int)layer_major_on());
        wk.auto_prompt_chunk=chosen;
        struct AutoScope{~AutoScope(){g_auto_dispatch=false;g_auto_chunk=0;}}auto_scope;
        g_auto_dispatch=true;g_auto_chunk=chosen;
        if(layer_major_on() || nt<=chosen)
            return forward_prefill(gguf_path,g,pack_dir,tokens,logits_out,err,opt);
        // Chunk-major remains available as the exact same selected-chunk reference.
        std::vector<int32_t> history;
        if(base_pos>0){
            if(!opt.history || opt.history->size()<(std::size_t)base_pos){err="automatic chunk: prefix history missing";return false;}
            history.assign(opt.history->begin(),opt.history->begin()+base_pos);
        }
        history.insert(history.end(),tokens.begin(),tokens.end());logits_out.clear();
        for(std::size_t at=0;at<tokens.size();at+=chosen){
            const std::size_t count=std::min((std::size_t)chosen,tokens.size()-at);
            std::vector<int32_t> part(tokens.begin()+at,tokens.begin()+at+count);std::vector<float> row;
            auto co=opt;co.history=&history;co.want_logits=opt.want_logits&&at+count==tokens.size();
            co.dfly = at + count == tokens.size() ? opt.dfly : nullptr;
            co.logits_last_only=opt.logits_last_only;
            if (opt.hidden_out) {
                if (opt.hidden_all_rows || !opt.logits_last_only)
                    co.hidden_out = opt.hidden_out + at * n_embd;
                else if (at + count != tokens.size())
                    co.hidden_out = nullptr;
            }
            if(!forward_prefill(gguf_path,g,pack_dir,part,row,err,co)){logits_out.clear();return false;}
            // Serving may return logits directly on the GPU. Only a host caller
            // has a vector to move; the callee already published the device view.
            if(co.want_logits && !opt.device_logits_out) logits_out=std::move(row);
        }
        return true;
    }


    if(allow_borrow){
        if(!layer_major_on()){err="prefill borrow currently requires layer-major";return false;}
        const auto before=sc_pool.generation;
        if(!prefill_phase_transition(res,g,opt,nt>kGemmT||g_layer_major,nt,stream,err))return false;
        // The transition clears persistent scratch. Reacquire the positions and device metadata;
        // no earlier GPU computation in this call has consumed them.
        if(sc_pool.generation!=before){
        d_pos=sc_pool.get(Scratch::kPos,(std::size_t)nt*4);
        if(cudaMemcpy(d_pos.p,pos.data(),pos.size()*4,cudaMemcpyHostToDevice)!=cudaSuccess){err="prefill borrow: position upload failed";return false;}
        d_kv_pos=sc_pool.get(Scratch::kKvPos,(std::size_t)kv_stride*4);
        if(!res->kv_pos_filled){std::vector<int32_t> kp(kv_stride);for(int i=0;i<kv_stride;++i)kp[i]=i;
            if(cudaMemcpy(d_kv_pos.p,kp.data(),kp.size()*4,cudaMemcpyHostToDevice)!=cudaSuccess){err="prefill borrow: KV positions failed";return false;}res->kv_pos_filled=true;}
        d_len=sc_pool.get(Scratch::kLen,8);const int32_t meta[2]={base_pos,base_pos+nt};
        if(cudaMemcpy(d_len.p,meta,8,cudaMemcpyHostToDevice)!=cudaSuccess){err="prefill borrow: metadata upload failed";return false;}
        }
    }
    // Allocate only after dense/tier setup and the prefill loan transition. The outer owner
    // survives scratch relayouts, layer boundaries and small tails; no captured decode uses it.
    if(g_layer_major && g_layer_major->state_device){
        auto& lm=*g_layer_major;
        if(!lm.state_storage.p)lm.state_storage=Buf(lm.state_bytes);
        if(!lm.state_storage.p){err="layer-major: device activation buffer allocation failed";return false;}
        lm.state=lm.state_storage.as<float>()+lm.state_offset;
    }
    // ---- the embedding, broadcast into all hc streams
    S state = sc_pool.get(Scratch::kState, (std::size_t) n_embd * hc * nt * 4);
    if(g_layer_major && g_layer_major->layer>0){
        if(cudaMemcpyAsync(state.p,g_layer_major->state,(std::size_t)n_embd*hc*nt*4,g_layer_major->state_device?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice,stream)!=cudaSuccess){err="layer-major: state upload failed";return false;}
    } else {
        S sc = sc_pool.get(Scratch::kEmbedSc, embed_tokens_scratch_bytes(n_embd, nt));
        W te = take(gg, head_slot, budget, "token_embd.weight");
        if (te.p == nullptr || sc.p == nullptr) {
            err = "forward_prefill: cannot upload token_embd";
            return false;
        }
        embed_tokens(te.p, te.type, tokens.data(), n_embd, hc, nt, state.as<float>(), sc.p, stream);
        if (opt.picture) opt.picture->embed(state.as<float>(), base_pos, nt, stream);
        cudaStreamSynchronize(stream);
    }

    // ---- the Engram token map, for the blocks that have a table
    std::vector<int32_t> own_tmap;
    if (res != nullptr) {
        if (res->token_map.empty() && !engram_token_map_load(pack_dir, res->token_map, err)) return false;
    } else if (!engram_token_map_load(pack_dir, own_tmap, err)) {
        return false;
    }
    const std::vector<int32_t>& tmap = res != nullptr ? res->token_map : own_tmap;
    const TensorInfo* table = gg.info("per_layer_token_embd.weight");
    if (table == nullptr) {
        err = "forward_prefill: the Engram table is missing from the model";
        return false;
    }
    const uint64_t table_base = gg.data_start() + table->offset;
    const int row_bytes = table->type == 43 ? g.engram_head_dim : g.engram_head_dim / 32 * 34;
    if (table->type != 8 && table->type != 43) {
        err = "forward_prefill: unsupported Engram encoding"; return false;
    }
    const int head_dim = g.engram_head_dim, columns = g.engram_heads;

    // Collect Engram row requests for every table before running the first block.
    auto engram_ids_for = [&](int il, std::vector<int64_t>& ids, std::vector<float>& valid,
                              std::string& e) -> bool {
        const LayerSpec& sp = g.layers[(std::size_t) il];
        const EngramTable& et = g.engram_tables[(std::size_t) sp.engram_table];
        EngramHashParams hp;
        hp.ngram_size = g.engram_ngram_size;
        hp.heads_per_ngram = g.engram_heads_per_ngram;
        hp.columns = columns;
        hp.pad_token = g.engram_pad_token;
        hp.multipliers = et.multipliers;
        hp.primes = et.head_bucket_counts;
        hp.offsets = et.head_offsets;
        hp.token_map = tmap.data();
        hp.token_map_len = (int64_t) tmap.size();
        // Resolve Engram predecessors by absolute position across chunk boundaries.
        const int64_t n_prev = hp.ngram_size - 1;
        std::vector<int32_t> prev((std::size_t) (nt * n_prev), 0);
        for (int i = 0; i < nt; ++i)
            for (int64_t sp2 = 0; sp2 < n_prev; ++sp2) {
                const int64_t back = n_prev - sp2;                   // how far behind this token
                const int64_t abs = (int64_t) base_pos + i - back;   // its absolute position
                int32_t id = 0;
                if (abs >= 0) {
                    if (abs >= base_pos)
                        id = tokens[(std::size_t) (abs - base_pos)];
                    else if (opt.history != nullptr && abs < (int64_t) opt.history->size())
                        id = (*opt.history)[(std::size_t) abs];
                }
                prev[(std::size_t) (i * n_prev + sp2)] = id;
            }
        const uint8_t* mask = nullptr;
        if (opt.picture && base_pos + nt <= static_cast<int>(opt.picture->image_mask.size()))
            mask = opt.picture->image_mask.data() + base_pos;
        return engram_row_ids(hp, tokens.data(), pos.data(), prev.data(), mask, nt, ids, valid, e);
    };
    // Read the four blocks' rows concurrently before they are consumed.
    std::vector<std::vector<float>> engram_rows((std::size_t) n_blocks);
    std::vector<std::vector<float>> engram_valid((std::size_t) n_blocks);
    {
        struct Job { int il; int t; int c; int64_t id; };
        std::vector<Job> jobs;
        bool ids_ok = true;
        for (int il = 0; il < n_blocks && ids_ok; ++il) {
            if(g_layer_major && il!=g_layer_major->layer)continue;
            if (g.layers[(std::size_t) il].engram_table < 0) continue;
            std::vector<int64_t> ids;
            if (!engram_ids_for(il, ids, engram_valid[(std::size_t) il], err)) {
                ids_ok = false;
                break;
            }
            engram_rows[(std::size_t) il].assign((std::size_t) head_dim * columns * nt, 0.0f);
            for (int t = 0; t < nt; ++t)
                for (int c = 0; c < columns; ++c)
                    jobs.push_back({il, t, c, ids[(std::size_t) (t * columns + c)]});
        }
        if (!ids_ok) return false;
        const auto t_pf = std::chrono::steady_clock::now();
        std::atomic<bool> failed{false};
        // One row's 102 bytes -> `head_dim` floats, wherever the bytes came from.
        const auto dequant_job = [&](std::size_t i, const uint8_t* src) {
            const Job& j = jobs[i];
            float* dst = engram_rows[(std::size_t) j.il].data() +
                (std::size_t) (j.t * columns + j.c) * head_dim;
            if (table->type == 43) dequant_f8_e4m3_row(src, head_dim, dst);
            else dequant_q8_0_row(src, head_dim, dst);
        };

        // Use the persistent direct-I/O row reader.
        bool read = false;
        if (res != nullptr && !jobs.empty() && engram_io_mode() == EngramIo::Direct) {
            if (!wk.engram) {
                wk.engram.reset(new EngramRows);
                std::string e;
                const uint64_t n_table_rows = table->shape.size() > 1 ? table->shape[1] : 0;
                std::string table_path;
                uint64_t physical_base = 0;
                if (!gg.physical(table_base, (std::size_t) tensor_payload_bytes(*table), table_path, physical_base))
                    e = "Engram table is outside the GGUF shards";
                else if (!wk.engram->open(table_path, physical_base, n_table_rows, (uint32_t) row_bytes, e))
                    e = "EngramRows: " + e;
                if (!e.empty()) {
                    std::fprintf(stderr, "hy4_flash: %s\n", e.c_str());
                    wk.engram.reset();
                }
            }
            if (wk.engram && wk.engram->is_open()) {
                std::vector<int64_t> want_ids(jobs.size());
                for (std::size_t i = 0; i < jobs.size(); ++i) want_ids[i] = jobs[i].id;
                wk.engram_raw.resize(jobs.size() * (std::size_t) row_bytes);
                std::string e;
                if (!wk.engram->gather(want_ids.data(), want_ids.size(), wk.engram_raw.data(), e)) {
                    err = "forward_prefill: reading the Engram rows: " + e;
                    return false;
                }
                for (std::size_t i = 0; i < jobs.size(); ++i)
                    dequant_job(i, wk.engram_raw.data() + i * (std::size_t) row_bytes);
                read = true;
            }
        }

        // Read from the GGUF mapping so repeated rows can be served from the page cache.
        if (!read) {
            static const int cap = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_ENGRAM_THREADS");
                const int n = v != nullptr ? std::atoi(v) : 16;
                return n > 0 ? n : 16;
            }();
            Pool* pool = nullptr;
            if (res != nullptr) {
                if (!wk.engram_pool) wk.engram_pool.reset(new Pool(cap));
                pool = wk.engram_pool.get();
            }
            const auto one = [&](std::size_t i) {
                if (failed.load()) return;
                std::vector<uint8_t> raw;
                const uint64_t at = table_base + (uint64_t) jobs[i].id * (uint64_t) row_bytes;
                const void* src = gg.at(at, (std::size_t) row_bytes);
                if (src == nullptr) {
                    // No mapping: a positioned read, which is thread-safe where fseek+fread is not.
                    raw.resize((std::size_t) row_bytes);
                    if (!gg.pread_at(at, raw.data(), raw.size())) {
                        failed.store(true);
                        return;
                    }
                    src = raw.data();
                }
                dequant_job(i, static_cast<const uint8_t*>(src));
            };
            if (pool != nullptr)
                pool->run((int) jobs.size(), [&](int i) { one((std::size_t) i); });
            else
                for (std::size_t i = 0; i < jobs.size(); ++i) one(i);
        }
        if (failed.load()) {
            err = "forward_prefill: reading the Engram rows";
            return false;
        }
        if (pt != nullptr) tick(pt->engram_read, t_pf);
        // S1: the same interval, with no CUDA call in it, so a graph-capturing run reports it too.
        if (!jobs.empty()) {
            g_engram_secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf).count();
            ++g_engram_steps;
            g_engram_requests += (long) jobs.size();
            if (engram_stats_on())
                for (const Job& j : jobs)
                    if (!g_engram_seen.insert((uint64_t) j.id).second) ++g_engram_recur;
        }
    }

    // Upload all four Engram inputs before any GPU expert wait can execute. Each table owns a
    // stable buffer: no default-stream pageable memcpy occurs between asynchronous MoE graphs.
    if(async_option || verify_prepare) {
        if(!wk.slices.drain_async()){err="async Engram upload drain";return false;}
        if(wk.async_engram_rows.size()!=(std::size_t)n_blocks) {
            wk.async_engram_rows.resize(n_blocks);wk.async_engram_valid.resize(n_blocks);
            wk.async_engram_rows_cap.resize(n_blocks);wk.async_engram_valid_cap.resize(n_blocks);
        }
        for(int ei=0;ei<n_blocks;++ei)if(!engram_rows[ei].empty()) {
            const std::size_t rb=engram_rows[ei].size()*sizeof(float),vb=engram_valid[ei].size()*sizeof(float);
            if(wk.async_engram_rows_cap[ei]<rb || wk.async_engram_valid_cap[ei]<vb) {
                // A changed reservation must never leave captured pointers to an old buffer.
                wk.invalidate_token_graph();
                wk.invalidate_window_graphs();
                wk.invalidate_verify_graphs();
                for(auto& graph:wk.attn_graph)if(graph){cudaGraphExecDestroy(graph);graph=nullptr;}
                if(wk.async_engram_rows_cap[ei]<rb){wk.async_engram_rows[ei]=Buf(rb);wk.async_engram_rows_cap[ei]=rb;}
                if(wk.async_engram_valid_cap[ei]<vb){wk.async_engram_valid[ei]=Buf(vb);wk.async_engram_valid_cap[ei]=vb;}
            }
            if(!wk.async_engram_rows[ei].p || !wk.async_engram_valid[ei].p ||
               cudaMemcpy(wk.async_engram_rows[ei].p,engram_rows[ei].data(),rb,cudaMemcpyHostToDevice)!=cudaSuccess ||
               cudaMemcpy(wk.async_engram_valid[ei].p,engram_valid[ei].data(),vb,cudaMemcpyHostToDevice)!=cudaSuccess){
                err="async Engram preupload failed";return false;
            }
        }
    }

    S scratch_hc = sc_pool.get(Scratch::kHcSc, hc_scratch_bytes(n_embd * hc, nt));
    S mixes = sc_pool.get(Scratch::kMixes, (std::size_t) 2 * hc * nt * 4);
    S pre = sc_pool.get(Scratch::kPre, (std::size_t) hc * nt * 4);
    S post = sc_pool.get(Scratch::kPost, (std::size_t) hc * nt * 4);
    S reduced = sc_pool.get(Scratch::kReduced, (std::size_t) n_embd * nt * 4);
    S next = sc_pool.get(Scratch::kNext, (std::size_t) n_embd * hc * nt * 4);
    S attn_out = sc_pool.get(Scratch::kAttnOut, (std::size_t) n_embd * nt * 4);
    S ffn_out = sc_pool.get(Scratch::kFfnOut, (std::size_t) n_embd * nt * 4);
    const bool window_option = expert_window_enabled();
    if(window_option && (pdl_on || !async_option ||
       g.top_k!=kWindowTopK || g.idx_kpool!=4)) {
        err="hy4-flash window path requires ASYNC_MOE=1, PDL=0, top16/kpool4";
        return false;
    }
    const bool async_decode=!(window_option && opt.speculative && opt.decode_equivalent) && !g_layer_major && async_option && wk.async_primed && nt==1 && base_pos>0 && !wk.slices.complete && opt.on_routing==nullptr && opt.on_block==nullptr && opt.teacher_state==nullptr && pt==nullptr && !opt.poison_scratch;
    const bool window_decode=!opt.picture && window_option && opt.speculative && opt.decode_equivalent &&
        nt>=1 && nt<=kWindowTokens && !g_layer_major && res && opt.cache && base_pos>0 &&
        !wk.slices.complete && !opt.on_routing && !opt.on_block &&
        !opt.teacher_state && !pt && !opt.poison_scratch;
    if(window_decode && !wk.slices.prepare_window(gg,nt,g.expert_dim,g.expert_ff,g.swiglu_clamp)) {
        err="window supply initialization failed (requires 32 staging slots per pool)";return false;
    }
    struct WindowPassGuard {
        ExpertSlices* slices=nullptr;cudaStream_t stream=nullptr;bool complete=false;
        ~WindowPassGuard() {
            if(slices&&!complete) {
                slices->window_fail("window pass abandoned");
                // Any nested capture guard has already ended capture at this point.
                cudaStreamSynchronize(stream);
            }
        }
    } window_guard{window_decode?&wk.slices:nullptr,stream};
    std::vector<ExpertAsyncGroup> async_groups((std::size_t)n_blocks);
    if(async_decode) {
        for(int ai=0;ai<n_blocks;++ai)if(wk.slices.pool_of[(std::size_t)ai]>=0)
            if(!wk.slices.prepare_async(gg,blk(ai),ai,n_embd,0,g.top_k,g.swiglu_clamp,&async_groups[(std::size_t)ai],stream)){
                err="async MoE initialization failed at block "+std::to_string(ai);return false;
            }
        for(int ai=0;ai<n_blocks;++ai)if(wk.slices.pool_of[(std::size_t)ai]>=0)
            async_groups[(std::size_t)ai].single_phase=single_phase;
    }
    int async_last_layer=-1;uint32_t async_last_expected=0;
    ExpertSlices& slices = wk.slices;   // kept across passes: its pinned ring is expensive to allocate

    // Keep indexer pools alongside the KV cache for the same conversation.
    static const bool idx_on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_INDEXER");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (idx_on && opt.cache != nullptr && g.idx_key_dim > 0 && g.idx_kpool > 0 && wk.idx == nullptr) {
        std::vector<int32_t> has((std::size_t) n_blocks, 0);
        for (int bi = 0; bi < n_blocks; ++bi)
            has[(std::size_t) bi] = g.layers[(std::size_t) bi].attention == Attention::Sliding ? 0 : 1;
        auto c = std::make_unique<IndexerCache>();
        if (!indexer_cache_init(*c, has, g.idx_key_dim, g.idx_kpool, (int) kv_stride, err)) return false;
        wk.idx = std::move(c);
    }
    // ---- THE OPT-IN SPECULATIVE-VERIFY CONTRACT (`ForwardOptions::dfly`, `exact_columns`). Everything is
    // refused here, before the pass writes anything, with a message that names the unmet condition.
    if (opt.dfly != nullptr) opt.dfly->valid = false;
    if (opt.dfly != nullptr || opt.exact_columns) {
        if (opt.cache == nullptr) { err = "dfly: needs a KV cache"; return false; }
        if (g_layer_major && (opt.exact_columns || opt.dfly->mode != DflyCapture::Mode::prefill_tail)) {
            err = "dfly: a verification window cannot execute inside a layer-major prefill";
            return false;
        }
    }
    if (opt.exact_columns) {
        if (nt < 1 || nt > DflyDrafter::kMaxK + 1 || nt > gemm_t()) {
            err = "dfly exact_columns: unsupported verification width (n_tok " + std::to_string(nt) +
                  ", gemm_t " + std::to_string(gemm_t()) + ")";
            return false;
        }
        if (wk.idx != nullptr &&
            indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, 1).engaged !=
                indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, nt).engaged) {
            err = "dfly exact_columns: the window crosses the indexer's engaged boundary at " +
                  std::to_string(base_pos) + "; shorten it";
            return false;
        }
    }
    // Each verify query needs its own visible key count and group-to-head mapping.
    static const bool pq_merged_env = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_VERIFY_PQ_MERGED");
        return v != nullptr && std::atoi(v) != 0;
    }();
    const bool pq_crosses = wk.idx != nullptr &&
        indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, 1).engaged !=
            indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, nt).engaged;
    const bool pq_window = opt.cache != nullptr && kv_format(*opt.cache) != kKvF32 && nt >= 2 &&
        (opt.exact_columns || (nt <= 4 && opt.decode_equivalent && pq_merged_env && !pq_crosses));
    if (opt.decode_equivalent && pq_merged_env && opt.cache != nullptr && kv_format(*opt.cache) != kKvF32 && nt >= 2 && nt <= 4) {
        static int said = 0;
        if ((said & (pq_window ? 1 : 2)) == 0) {
            said |= pq_window ? 1 : 2;
            std::fprintf(stderr, "HY4_FLASH_VERIFY_PQ_MERGED %s: n_tok=%d base_pos=%d\n",
                         pq_window ? "engaged" : "declined (crosses the indexer's engaged boundary) -> per-head path",
                         nt, base_pos);
        }
    }
    MlaPerQuery pq_meta{};
    int32_t* pq_dev = nullptr;
    int pq_heads = 0;
    if (pq_window) {
        for (const auto& l : g.layers) pq_heads = std::max(pq_heads, (int) l.n_heads);
        S d_meta = sc_pool.get(Scratch::kExactMeta, ((std::size_t) nt + (std::size_t) pq_heads * nt) * 4);
        if (d_meta.p == nullptr) {
            err = "verify window: the per-query metadata allocation failed";
            return false;
        }
        pq_dev = d_meta.as<int32_t>();
        pq_meta.n_kv = pq_dev;
        pq_meta.head_of = pq_dev + nt;
    }
    if (opt.dfly != nullptr) {
        const DflyCapture& d = *opt.dfly;
        const bool verify = d.mode == DflyCapture::Mode::verify_window;
        if ((!opt.want_logits && !g_layer_major) || (verify && opt.logits_last_only)) {
            err = "dfly: capture needs logits, and every row of them for a verify window";
            return false;
        }
        if (wk.idx == nullptr || d.hidden == nullptr || d.n_embd != n_embd || d.key_dim != g.idx_key_dim ||
            d.kpool != g.idx_kpool || d.n_idx_layers != wk.idx->n_layers ||
            (std::size_t) d.n_idx_layers != d.vg.size() || d.sel.size() != d.sel_blocks.size()) {
            err = "dfly: the capture is not initialised for this model, or the indexer is off";
            return false;
        }
        if (verify && (nt > d.capacity_rows || d.kpool - 1 + nt > d.kpool - 1 + d.capacity_rows)) {
            err = "dfly: a " + std::to_string(nt) + "-row window exceeds the capture's " +
                  std::to_string(d.capacity_rows) + " rows";
            return false;
        }
        for (int b : d.sel_blocks)
            if (b < 0 || b >= n_blocks || wk.idx->ids[(std::size_t) b] < 0) {
                err = "dfly: selection block " + std::to_string(b) + " is not a sparse block";
                return false;
            }
    }
    // Reset each layer's pooled state when the next pass resumes from a different position.
    if(g_layer_major && wk.idx){
        wk.idx->n_pool=base_pos/g.idx_kpool;wk.idx->carry=base_pos%g.idx_kpool;wk.idx->valid_pos=base_pos;
        const int counters[2]={wk.idx->carry,wk.idx->n_pool};
        if(cudaMemcpyAsync(wk.idx->counters,counters,sizeof(counters),cudaMemcpyHostToDevice,stream)!=cudaSuccess){err="layer-major: indexer cursor upload failed";return false;}
    }
    if (wk.idx != nullptr && !indexer_cache_rewind(*wk.idx, base_pos)) {
        err = "indexer cache cannot rewind from " + std::to_string(wk.idx->valid_pos) + " to " +
              std::to_string(base_pos) + " (mid-group); align reuse to " + std::to_string(g.idx_kpool);
        return false;
    }
    if (wk.idx && opt.speculative) {
        auto& idx = *wk.idx;
        if (!idx.verify_capacity) {
            idx.verify_capacity = 8;
            idx.verify_vg.assign(idx.n_layers, nullptr);
            for (void*& ptr : idx.verify_vg)
                if (cudaMalloc(&ptr, (std::size_t) 2 * idx.key_dim * (8 + idx.kpool) * sizeof(float)) != cudaSuccess) {
                    err = "speculative indexer checkpoint allocation failed"; return false;
                }
        }
        idx.verify_base = nt <= idx.verify_capacity ? base_pos : -1;
        idx.verify_tokens = nt;
    }

    // Use GEMM for eligible prompt chunks while keeping decode on the matvec path.
    MatCtx* mc = nullptr;
    if (nt > gemm_t()) {
        // The head count is per block (LayerSpec::n_heads, not a scalar), so take the widest block.
        std::int64_t heads = 0;
        for (const LayerSpec& ls : g.layers) heads = std::max<std::int64_t>(heads, ls.n_heads);
        const std::int64_t widest = std::max<std::int64_t>({(std::int64_t) n_embd * hc,
                                                            (std::int64_t) g.q_lora_rank,
                                                            (std::int64_t) g.v_head_dim * heads,
                                                            (std::int64_t) g.dense_ff});
        const std::int64_t welems =
            std::max<std::int64_t>((std::int64_t) g.dense_ff * n_embd,
                                   (std::int64_t) g.q_lora_rank * g.qk_head_dim * heads);
        std::string merr;
        mc = mat_ctx(nt, widest, welems, stream, merr);
        if((auto_chunk_on() || allow_borrow) && !mat_ctx_require_prefill(mc,stream,merr)){
            err=merr;return false;
        }
        if (mc == nullptr) {
            // Fall back to split matvecs if the GEMM workspace cannot be allocated.
            static int said = 0;
            if (said++ == 0)
                std::fprintf(stderr, "hy4_flash: the prompt's GEMM backend could not be built for %d-token "
                                     "chunks, so they run on the matvecs - %s\n", nt, merr.c_str());
        } else {
            mat_pass_begin(mc);
        }
    }

    // Read pooling extents from device counters so captured launches can replay at different positions.
    const bool idx_engaged =
        wk.idx != nullptr && opt.cache != nullptr &&
        indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, nt).engaged;
    if (wk.idx != nullptr && idx_engaged != wk.idx_engaged) {
        wk.idx_engaged = idx_engaged;
        wk.invalidate_window_graphs();
        wk.invalidate_token_graph();
        for (std::size_t bi = 0; bi < wk.attn_graph.size(); ++bi)
            if (wk.attn_graph[bi] != nullptr) {
                cudaGraphExecDestroy(wk.attn_graph[bi]);
                wk.attn_graph[bi] = nullptr;
            }
        std::fprintf(stderr, "hy4_flash indexer: selection %s at position %d - graphs rebuilt\n",
                     idx_engaged ? "ENGAGED" : "disengaged", base_pos);
    }

    static const bool token_graph_option = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_TOKEN_GRAPH");
        return v && std::atoi(v) != 0;
    }();
    // Only the mature partial-resident async T1 path is covered initially. Host observers,
    // prefill, no-cache and Graph0 retain the original implementation.
    static const bool window_graph_option=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_GRAPH");return e&&std::atoi(e)!=0;}();
    const bool window_graph_path=window_decode && window_graph_option && graphs_enabled() &&
        wk.warm && opt.want_logits && !opt.logits_last_only && opt.dfly == nullptr;
    const bool token_path = window_graph_path || (!opt.picture && !opt.speculative && token_graph_option && graphs_enabled() && async_decode &&
                            opt.dfly == nullptr && res != nullptr && opt.cache != nullptr && opt.want_logits && wk.warm);
    const int head_rows = opt.hidden_all_rows ? nt : opt.logits_last_only ? 1 : nt;
    const int logit_rows = opt.want_logits ? (opt.logits_last_only ? 1 : nt) : 0;
    S d_logits;
    if (token_path && !window_graph_path) {
        // All of these addresses were exercised by a complete eager T1 pass before capture.
        d_logits = sc_pool.get(Scratch::kLogits, (std::size_t)vocab * head_rows * 4);
        const bool changed = wk.token_scratch_generation != sc_pool.generation ||
                             wk.token_kv_stride != kv_stride || wk.token_kv_format != kv_format(*opt.cache) ||
                             wk.token_kv != opt.cache->kv_cmpr || wk.token_pe != opt.cache->k_pe;
        if (changed) {
            wk.invalidate_token_graph();
            wk.token_scratch_generation = sc_pool.generation;
            wk.token_kv_stride = kv_stride;
            wk.token_kv_format = kv_format(*opt.cache);
            wk.token_kv = opt.cache->kv_cmpr;
            wk.token_pe = opt.cache->k_pe;
        }
    }
    if(window_graph_path) {
        d_logits=sc_pool.get(Scratch::kLogits,(std::size_t)vocab*logit_rows*4);
        if(!wk.window_hidden.p)wk.window_hidden=Buf((std::size_t)kWindowTokens*n_embd*4);
        if(!wk.window_hidden.p){err="window hidden scratch allocation failed";return false;}
        if(wk.window_generation!=sc_pool.generation || wk.window_kv_stride!=kv_stride ||
           wk.window_kv_format!=kv_format(*opt.cache) || wk.window_kv!=opt.cache->kv_cmpr || wk.window_pe!=opt.cache->k_pe) {
            wk.invalidate_window_graphs();wk.window_generation=sc_pool.generation;
            wk.window_kv_format=kv_format(*opt.cache);
            wk.window_kv_stride=kv_stride;wk.window_kv=opt.cache->kv_cmpr;wk.window_pe=opt.cache->k_pe;
        }
    }
    auto& wg=wk.window_graph[window_graph_path?nt:0][window_graph_path?base_pos%4:0];
    cudaGraphExec_t& pass_graph=window_graph_path?wg.exec:wk.token_graph;
    bool& pass_warm=window_graph_path?wg.warm:wk.token_warm;
    // The fallback attention loop bakes each query's host length. Capture only geometries that
    // use the merged, device-length implementation in every block.
    bool verify_geometry = verify_prepare;
    if (verify_geometry && nt > 1) {
        for (int il = 0; il < n_blocks; ++il) {
            const auto* kb = gg.info(blk(il) + "attn_k_b.weight");
            const auto* vb = gg.info(blk(il) + "attn_v_b.weight");
            if (!kb || !vb || !mla_per_query_supported(g.kv_lora_rank, g.rope_dim,
                    g.layers[il].n_heads, opt.cache->fp16, (int)kb->type, (int)vb->type)) {
                verify_geometry = false;
                break;
            }
        }
    }
    // ASYNC_MOE enables offload when the tier is partial; a complete tier never
    // enters that executor. Admit graphs from the actual execution path, since
    // the serving environment also enables ASYNC_MOE on fully resident models.
    const bool verify_path = verify_geometry && wk.warm && wk.slices.complete &&
                             !async_decode && !window_decode &&
                             !opt.logits_last_only && opt.want_logits;
    DenseResidency::Work::VerifyGraph* verify_entry = nullptr;
    if (verify_path) {
        d_logits = sc_pool.get(Scratch::kLogits, (std::size_t)vocab * nt * 4);
        // Below engagement, plan.n_sel still grows with position. Reserve its ceiling before capture.
        const int max_sel = g.idx_top_k + g.idx_kpool;
        if (!sc_pool.get(Scratch::kIndexerSc, indexer_scratch_bytes(g.idx_heads, g.idx_key_dim,
                g.idx_kpool, nt, wk.idx->max_pool, max_sel)).p ||
            !sc_pool.get(Scratch::kIndexerSel, (std::size_t)max_sel * nt * 4).p) {
            err = "dfly graph: indexer scratch allocation failed";
            return false;
        }
        std::vector<const void*> addresses{opt.cache, wk.idx.get(), opt.dfly->hidden,
                                          wk.idx ? wk.idx->counters : nullptr};
        for (const auto& v : {opt.cache->kv_cmpr, opt.cache->k_pe, wk.idx->pooled, wk.idx->vg,
                             opt.dfly->sel, opt.dfly->vg})
            addresses.insert(addresses.end(), v.begin(), v.end());
        for (const auto& b : wk.async_engram_rows) addresses.push_back(b.p);
        for (const auto& b : wk.async_engram_valid) addresses.push_back(b.p);
        std::vector<int> shape{kv_stride, kv_format(*opt.cache), opt.dfly->capacity_rows,
                              opt.dfly->sel_cap, opt.dfly->n_embd, opt.dfly->key_dim, opt.dfly->kpool};
        shape.insert(shape.end(), opt.dfly->sel_blocks.begin(), opt.dfly->sel_blocks.end());
        if (addresses != wk.verify_addresses || shape != wk.verify_shape ||
            wk.verify_scratch_generation != sc_pool.generation) {
            wk.invalidate_verify_graphs();
            wk.verify_addresses = std::move(addresses);
            wk.verify_shape = std::move(shape);
            wk.verify_scratch_generation = sc_pool.generation;
        }
        verify_entry = &wk.verify_graphs[(nt - 1) * 2 + (idx_engaged ? 1 : 0)];
    }
    const bool verify_replay = verify_entry && verify_entry->exec;
    const bool verify_capturing = verify_entry && verify_entry->warm && !verify_replay;
    const bool token_replay = (token_path && pass_graph != nullptr) || verify_replay;
    const bool token_capturing = (token_path && pass_warm && !token_replay) || verify_capturing;
    struct TokenCaptureGuard {
        cudaStream_t stream;
        bool active = false;
        ~TokenCaptureGuard() {
            if (active) {
                cudaGraph_t abandoned = nullptr;
                cudaStreamEndCapture(stream, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
            }
        }
    } token_capture_guard{stream};
    std::chrono::steady_clock::time_point capture_start;
    if (token_capturing) {
        capture_start = std::chrono::steady_clock::now();
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            err = "forward_prefill: token graph begin capture failed";
            return false;
        }
        token_capture_guard.active = true;
    }
    // The requested epoch is read BEFORE launch, never inferred from a temporarily idle worker.
    int window_last=-1;uint32_t window_epoch=0;
    if(window_decode)for(int il=0;il<n_blocks;++il)if(wk.slices.pool_of[il]>=0) {
        window_last=il;window_epoch=ExpertSlices::flag_load(&wk.slices.window_exec.layers[il]->host->request)+1;
    }
    if (token_path && !window_graph_path) {
        for (int il=0; il<n_blocks; ++il) if (wk.slices.pool_of[(std::size_t)il]>=0) {
            async_last_layer=il;
            async_last_expected=ExpertSlices::flag_load(&wk.slices.async_layers[(std::size_t)il]->host->request)+1;
        }
    }
    auto t_hd = std::chrono::steady_clock::now();
    if (!token_replay) {
    // Inside any whole-pass capture that began above, so a replay recomputes the counts from its own d_len.
    if (pq_dev != nullptr) mla_per_query_meta(d_len.as<int32_t>(), pq_dev, pq_dev + nt, nt, pq_heads, stream);
    const int first_layer=g_layer_major?g_layer_major->layer:0;
    const int end_layer=g_layer_major?first_layer+1:n_blocks;
    for (int il = first_layer; il < end_layer; ++il) {
        // Reuse the FP16 activation arena after each block finishes reading its inputs.
        mat_pass_begin(mc);
        const LayerSpec& spec = g.layers[(std::size_t) il];
        const uint32_t window_expected=window_decode && wk.slices.pool_of[il]>=0 ?
            ExpertSlices::flag_load(&wk.slices.window_exec.layers[il]->host->request)+1:0;
        const bool async_this_layer=async_decode && wk.slices.pool_of[(std::size_t)il]>=0;
        const uint32_t async_expected=async_this_layer ? ExpertSlices::flag_load(&wk.slices.async_layers[(std::size_t)il]->host->request)+1 : 0;
        DenseResidency::Slot* rs = res != nullptr ? res->slots[(std::size_t) il] : nullptr;
        const bool sliding = spec.attention == Attention::Sliding;
        const float rope_base = sliding ? g.rope_theta_sliding : g.rope_theta_sparse;
        // 512 attended positions INCLUDING the query, matching is_masked_swa's p1 - p0 >= n_swa.
        const int window = sliding ? g.sliding_window : 0;
        // The ceiling on the indexer's per-query cell count, for the attention scratch's slot list. From the
        // GEOMETRY, so it is the same for every block and every chunk - `indexer_plan` can only return
        // `kpool * n_top + kpool` with `n_top <= top_k/kpool - 1`, which is `top_k` once engaged.
        const int max_sel =
            (!sliding && wk.idx != nullptr && g.idx_top_k > 0) ? g.idx_top_k + g.idx_kpool : 0;

        // --- Engram first, where the reference puts it: it writes into the streams BEFORE the attention
        // pre-mix reads them, so a table block's input is engram(previous l_out).
        if (spec.engram_table >= 0) {
            // Already read, above, together with every other table block's rows.
            const std::vector<float>& rows = engram_rows[(std::size_t) il];
            const std::vector<float>& valid = engram_valid[(std::size_t) il];
            const auto t_eg = std::chrono::steady_clock::now();
            S d_rows,d_valid;
            if(async_option || verify_prepare){d_rows.p=wk.async_engram_rows[il].p;d_valid.p=wk.async_engram_valid[il].p;}
            else {
                d_rows=sc_pool.get(Scratch::kEngramRows, rows.size()*4);
                d_valid=sc_pool.get(Scratch::kEngramValid, valid.size()*4);
                if (!d_rows.p || !d_valid.p ||
                    cudaMemcpy(d_rows.p,rows.data(),rows.size()*4,cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(d_valid.p,valid.data(),valid.size()*4,cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "forward_prefill: cannot upload Engram inputs for " + std::to_string(nt) + " tokens";
                    return false;
                }
            }
            W e_kv = take(gg, rs, budget, blk(il) + "engram_kv.weight");
            W e_kvb = take(gg, rs, budget, blk(il) + "engram_kv_b.weight");
            W e_gate = take(gg, rs, budget, blk(il) + "engram_gate.weight");
            W e_ln = take(gg, rs, budget, blk(il) + "engram_lookup_norm.weight");
            W e_pg = take(gg, rs, budget, blk(il) + "engram_pre_gamma.weight");
            W e_hs = take(gg, rs, budget, blk(il) + "engram_hc_scale.weight");
            const size_t engram_scratch = engram_layer_scratch_bytes(n_embd, hc, head_dim, columns, nt);
            S e_sc = sc_pool.get(Scratch::kEngramSc, engram_scratch);
            if (!e_sc.p) {
                size_t free_bytes = 0, total_bytes = 0;
                cudaMemGetInfo(&free_bytes, &total_bytes);
                cudaGetLastError();
                err = "forward_prefill: cannot allocate Engram buffers for " + std::to_string(nt) +
                      " tokens (scratch " + std::to_string(engram_scratch >> 20) + " MiB, " +
                      std::to_string(free_bytes >> 20) + " MiB free); lower the prefill chunk or --expert-vram";
                return false;
            }
            if (!e_kv.p || !e_kvb.p || !e_gate.p || !e_ln.p || !e_pg.p || !e_hs.p) {
                err = "forward_prefill: block " + std::to_string(il) + "'s Engram weights";
                return false;
            }
            EngramLayerWeights ew{};
            ew.kv = e_kv.p;
            ew.kv_type = e_kv.type;
            ew.kv_b = e_kvb.as<float>();
            ew.gate = e_gate.as<float>();
            ew.lookup_norm = e_ln.as<float>();
            ew.pre_gamma = e_pg.as<float>();
            ew.hc_scale = e_hs.as<float>();
            engram_layer(state.as<float>(), d_rows.as<float>(), d_valid.as<float>(), ew, n_embd, hc, head_dim,
                         columns, nt, next.as<float>(), e_sc.p, stream, mc);
            // The stream orders the kernels across the host pointer swap.
            std::swap(state, next);
            if (pt != nullptr) {
                cudaStreamSynchronize(stream);
                tick(pt->engram_gpu, t_eg);
            }
        }

        // --- the attention half
        const auto t_aw = std::chrono::steady_clock::now();
        W hcfn = take(gg, rs, budget, blk(il) + "hc_attn_fn.weight");
        W hcs = take(gg, rs, budget, blk(il) + "hc_attn_scale.weight");
        W hcb = take(gg, rs, budget, blk(il) + "hc_attn_base.weight");
        W anorm = take(gg, rs, budget, blk(il) + "attn_norm.weight");
        W qa = take(gg, rs, budget, blk(il) + "attn_q_a.weight");
        W qan = take(gg, rs, budget, blk(il) + "attn_q_a_norm.weight");
        W qb = take(gg, rs, budget, blk(il) + "attn_q_b.weight");
        W kvm = take(gg, rs, budget, blk(il) + "attn_kv_a_mqa.weight");
        W kvn = take(gg, rs, budget, blk(il) + "attn_kv_a_norm.weight");
        W kb = take(gg, rs, budget, blk(il) + "attn_k_b.weight");
        W vb = take(gg, rs, budget, blk(il) + "attn_v_b.weight");
        W sinks = take(gg, rs, budget, blk(il) + "attn_sinks.weight");
        W agate = take(gg, rs, budget, blk(il) + "attn_gate.weight");
        W wo = take(gg, rs, budget, blk(il) + "attn_output.weight");
        W pan = take(gg, rs, budget, blk(il) + "post_attention_norm.weight");
        // The compressed indexer's eight, on the sparse blocks only. A sliding block attends a 512 window and
        // needs no selection; a sparse block has no window and is what the indexer exists for.
        W ix_k, ix_qb, ix_pr, ix_kn, ix_cv, ix_cg, ix_cn, ix_ape;
        if (!sliding) {
            ix_k = take(gg, rs, budget, blk(il) + "indexer.attn_k.weight");
            ix_qb = take(gg, rs, budget, blk(il) + "indexer.attn_q_b.weight");
            ix_pr = take(gg, rs, budget, blk(il) + "indexer.proj.weight");
            ix_kn = take(gg, rs, budget, blk(il) + "indexer.k_norm.weight");
            ix_cv = take(gg, rs, budget, blk(il) + "indexer_compressor_value.weight");
            ix_cg = take(gg, rs, budget, blk(il) + "indexer_compressor_gate.weight");
            ix_cn = take(gg, rs, budget, blk(il) + "indexer_compressor_norm.weight");
            ix_ape = take(gg, rs, budget, blk(il) + "indexer_compressor_ape.weight");
        }
        if (!hcfn.p || !qa.p || !qb.p || !kvm.p || !kb.p || !vb.p || !agate.p || !wo.p || !pan.p) {
            // Include available VRAM when reporting allocation failures.
            std::size_t mfree = 0, mtot = 0;
            cudaMemGetInfo(&mfree, &mtot);
            cudaGetLastError();
            err = "forward_prefill: block " + std::to_string(il) + "'s attention weights did not fit (" +
                  std::to_string((long) (mfree / (1024 * 1024))) + " MiB free of " +
                  std::to_string((long) (mtot / (1024 * 1024))) + ", " + std::to_string(nt) +
                  "-token chunk) - lower STRATA_HY4_FLASH_PREFILL_CHUNK or --expert-vram";
            return false;
        }
        const int n_head = spec.n_heads;
        const int qk_head = (int) (qb.dim(1) / n_head);
        const int n_rot = g.rope_dim, kv_lora = g.kv_lora_rank, q_lora = g.q_lora_rank;
        const int v_head = (int) vb.dim(1);
        const int nope = qk_head - n_rot;
        if (pt != nullptr) tick(pt->weights, t_aw);

        const auto t_at = std::chrono::steady_clock::now();

        // Warm weights before capture and keep scratch pointers stable. KV position and length change on
        // device.
        static const bool verify_graph_option = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_VERIFY_GRAPH");
            return v == nullptr || std::atoi(v) != 0;
        }();
        static const int verify_graph_max_nt = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_VERIFY_GRAPH_MAX_NT");
            const int n = v != nullptr ? std::atoi(v) : 4;
            return n < 1 ? 1 : (n > kWindowTokens ? kWindowTokens : n);
        }();
        // Prompt chunks in a speculative request also set speculative. Only decode-equivalent
        // windows use this graph. The async option preloads Engram rows into stable per-layer
        // buffers so no blocking table upload occurs during capture.
        const bool spec_capture = verify_graph_option && opt.speculative && opt.decode_equivalent &&
                                  nt > 1 && nt <= verify_graph_max_nt && async_option &&
                                  wk.slices.complete && !window_decode && !async_decode &&
                                  wk.spec_warm_nt == nt;
        // Report the first declined window to explain why capture was skipped.
        if (verify_graph_option && opt.speculative && opt.decode_equivalent && !spec_capture && il == 0) {
            static int said = 0;
            if (said++ == 0)
                std::fprintf(stderr,
                             "HY4_FLASH_VERIFY_GRAPH declined: nt=%d complete=%d window=%d async=%d graphs=%d "
                             "warm=%d spec_warm_nt=%d max_nt=%d token_path=%d layer_major=%d cache=%d base_pos=%d\n",
                             nt, wk.slices.complete ? 1 : 0, window_decode ? 1 : 0, async_decode ? 1 : 0,
                             graphs_enabled() ? 1 : 0, wk.warm ? 1 : 0, wk.spec_warm_nt, verify_graph_max_nt, token_path ? 1 : 0,
                             g_layer_major ? 1 : 0, opt.cache != nullptr ? 1 : 0, base_pos);
        }
        const bool can_capture = !opt.picture && (!opt.speculative || spec_capture) && !g_layer_major && !token_path && opt.dfly == nullptr && res != nullptr && opt.cache != nullptr && (nt == 1 || spec_capture) && base_pos > 0 &&
                                 opt.on_block == nullptr && opt.teacher_state == nullptr &&
                                 opt.on_attn == nullptr &&
                                 opt.on_routing == nullptr && pt == nullptr && !opt.poison_scratch &&
                                 graphs_enabled() && wk.warm && (!async_option || wk.async_primed || spec_capture);
        // Capture attention alone when isolating FFN capture failures.
        static const bool capture_attn_only = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_CAPTURE_ATTN_ONLY");
            return v != nullptr && std::atoi(v) != 0;
        }();
        // Identify the first operation that invalidates stream capture.
        static const bool capture_probe = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_CAPTURE_PROBE");
            return v != nullptr && std::atoi(v) != 0;
        }();
        const bool partial_attn_graph = can_capture &&
            ((!wk.slices.complete && !async_decode) || (spec_capture && capture_attn_only));
        cudaGraphExec_t graph = can_capture ? wk.attn_graph[(std::size_t) il] : nullptr;
        const bool replay = graph != nullptr;
        const bool capturing = can_capture && !replay;
        auto probe = [&](const char* where) {
            if (!capture_probe || !capturing) return;
            cudaStreamCaptureStatus st{};
            if (cudaStreamGetCaptureInfo(stream, &st, nullptr) != cudaSuccess ||
                st != cudaStreamCaptureStatusActive) {
                static int said = 0;
                if (said++ == 0)
                    std::fprintf(stderr, "HY4_FLASH_CAPTURE_PROBE invalidated by nt=%d block=%d after %s\n",
                                 nt, il, where);
            }
        };
        if (capturing && cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            err = "forward_prefill: cannot begin graph capture at block " + std::to_string(il);
            return false;
        }
        if (!replay) {
        hc_pre(state.as<float>(), hcfn.p, hcs.as<float>(), hcb.as<float>(), n_embd, hc, nt, 2 * hc, rms_eps,
               hc_eps, magnitude, scratch_hc.p, mixes.as<float>(), pre.as<float>(), post.as<float>(),
               reduced.as<float>(), stream, mc, hcfn.type);
        // The norm and the one Q8 of it that the whole half shares, in one launch. `reduced` stays F32:
        // mla_output and the router read it as floats.
        S cur_q8 = sc_pool.get(Scratch::kCurQ8, kernels::native_q8_1_bytes_wide(n_embd, nt));
        if (!kernels::native_rms_norm_quantize_q8_1(reduced.as<float>(), anorm.as<float>(),
                                                    reduced.as<float>(), cur_q8.p, nt, n_embd, rms_eps,
                                                    stream)) {
            kernels::rms_norm_weighted_wide(reduced.as<float>(), anorm.as<float>(), nt, n_embd, rms_eps, stream);
            kernels::native_quantize_q8_1_wide(reduced.as<float>(), cur_q8.p, n_embd, nt, stream);
        }
        // The pair this half's projections read `reduced` through: the Q8_1 above, plus the FP16 image on a
        // prompt chunk.
        const MatIn cur_in = mat_staged(mc, reduced.as<float>(), cur_q8.p, n_embd, nt, stream);
        probe("hc_pre + norm + stage");

        S p_sc = sc_pool.get(Scratch::kProjSc,
                             mla_project_scratch_bytes(n_embd, q_lora, kv_lora, qk_head, n_rot, n_head, nt));
        S a_sc = sc_pool.get(Scratch::kAbsorbSc, mla_absorb_scratch_bytes(nope, nt, n_head));
        S absorbed = sc_pool.get(Scratch::kAbsorbed, (std::size_t) kv_lora * nt * n_head * 4);
        // Per-query verification uses each token's decode key count, window floor and
        // attention split to match the single-token reduction order.
        static const bool per_query_env = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_VERIFY_PER_QUERY");
            return v != nullptr && std::atoi(v) != 0;
        }();
        // Batch per-query verification projections while retaining single-token arithmetic.
        static const bool pq_loop_env = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_PQ_LOOP");
            return v != nullptr && std::atoi(v) != 0;
        }();
        const bool pq_merged = pq_meta.n_kv != nullptr && !pq_loop_env &&
                               mla_per_query_supported(kv_lora, n_rot, n_head, opt.cache->fp16, kb.type, vb.type);
        const bool per_query = pq_merged || ((per_query_env || opt.exact_columns) && opt.cache != nullptr &&
                                             nt >= 2 && nt <= kGemmT);
        // Sized by the RESERVED context, not by how long the sequence is now: a buffer that grows every
        // token is a buffer that moves, and a captured graph holds the pointer it saw.
        S at_sc = sc_pool.get(Scratch::kAttnSc,
                              std::max(mla_attention_scratch_bytes(kv_lora, v_head, n_head, nt, kv_stride,
                                                                   max_sel),
                                       pq_merged ? mla_attention_scratch_bytes(kv_lora, v_head, n_head, nt,
                                                                               kv_stride, max_sel, true)
                                       : per_query ? mla_attention_scratch_bytes(kv_lora, v_head, n_head, 1,
                                                                                 kv_stride, max_sel)
                                                   : (std::size_t) 0));
        S kqv = sc_pool.get(Scratch::kKqv, (std::size_t) v_head * n_head * nt * 4);
        S o_sc = sc_pool.get(Scratch::kOutSc, mla_output_scratch_bytes(n_embd, v_head * n_head, nt));
        // Compute the attention gate concurrently with the exact verification path.
        static const bool gate_fork_env = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_GATE_FORK");
            return v != nullptr && std::atoi(v) != 0;
        }();
        // One stream and two events per thread AND device: made on first use, made again when the thread's
        // device has changed, released at thread exit (errors ignored there - at process exit the runtime
        // may already be gone). A partial creation releases what it made.
        struct GateForkStream {
            cudaStream_t stream = nullptr;
            cudaEvent_t fork = nullptr, join = nullptr;
            int device = -1;
            void release() {
                if (join != nullptr) cudaEventDestroy(join);
                if (fork != nullptr) cudaEventDestroy(fork);
                if (stream != nullptr) cudaStreamDestroy(stream);
                stream = nullptr;
                fork = join = nullptr;
                device = -1;
            }
            bool ensure() {
                int dev = -1;
                if (cudaGetDevice(&dev) != cudaSuccess) return false;
                if (dev == device) return true;
                release();
                cudaStream_t s = nullptr;
                cudaEvent_t f = nullptr, j = nullptr;
                const bool made = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
                                  cudaEventCreateWithFlags(&f, cudaEventDisableTiming) == cudaSuccess &&
                                  cudaEventCreateWithFlags(&j, cudaEventDisableTiming) == cudaSuccess;
                stream = s;
                fork = f;
                join = j;
                if (!made) {
                    cudaGetLastError();
                    release();
                    return false;
                }
                device = dev;
                return true;
            }
            ~GateForkStream() { release(); }
        };
        static thread_local GateForkStream gate_fs;
        cudaEvent_t gate_join = nullptr;
        const bool gate_fork = gate_fork_env && opt.exact_columns && nt > 1 && cur_in.q8 != nullptr &&
                               !mat_gemm_path(cur_in.mc, nt);
        if (gate_fork) {
            if (!gate_fs.ensure()) {
                err = "forward_prefill: the gate fork's stream and events could not be created";
                return false;
            }
            if (cudaEventRecord(gate_fs.fork, stream) != cudaSuccess ||
                cudaStreamWaitEvent(gate_fs.stream, gate_fs.fork, 0) != cudaSuccess ||
                cudaGetLastError() != cudaSuccess) {
                err = "forward_prefill: the gate fork could not leave the main stream";
                return false;
            }
            mla_gate(agate.p, agate.type, cur_in, n_embd, v_head * n_head, nt, o_sc.p, gate_fs.stream);
            if (cudaEventRecord(gate_fs.join, gate_fs.stream) != cudaSuccess ||
                cudaGetLastError() != cudaSuccess) {
                err = "forward_prefill: the gate fork's join event failed";
                return false;
            }
            gate_join = gate_fs.join;
        }
        // wq_a, wkv_a_mqa and the attention gate all read the same `reduced`, which nothing between them
        // writes, so the one Q8 made with the norm above serves all three.
        MlaProjected pj{};
        mla_project(reduced.as<float>(), qa.p, qa.type, qan.as<float>(), qb.p, qb.type, kvm.p, kvm.type,
                    kvn.as<float>(), n_embd, q_lora, kv_lora, qk_head, n_rot, n_head, nt,
                    d_pos.as<int32_t>(), rms_eps, rope_base, p_sc.p, pj, cur_in, stream, opt.picture ? opt.picture->rope : ImageRope{});
        probe("mla_project");
        if (pq_merged) {
            // [kv_lora, nt, n_head], the batched layout, which is what the per-query attention reads.
            mla_absorb(pj.q, kb.p, kb.type, qk_head, nope, kv_lora, n_head, nt, absorbed.as<float>(), a_sc.p,
                       stream, &pq_meta);
        } else if (per_query) {
            for (int j = 0; j < nt; ++j)
                mla_absorb(pj.q + (std::size_t) j * qk_head * n_head, kb.p, kb.type, qk_head, nope, kv_lora,
                           n_head, 1, absorbed.as<float>() + (std::size_t) j * kv_lora * n_head, a_sc.p, stream);
        } else {
            mla_absorb(pj.q, kb.p, kb.type, qk_head, nope, kv_lora, n_head, nt, absorbed.as<float>(), a_sc.p,
                       stream);
        }
        probe("mla_absorb");
        // With a cache: append this chunk's rows, then attend over everything cached. Without: the chunk
        // attends itself.
        const void* keys = pj.kv_cmpr;
        const void* keys_pe = pj.k_pe;
        // Without a cache the rows are the projection's own, which are F32 whatever the cache would be.
        int keys_format = kKvF32;
        const int32_t* kv_pos = d_pos.as<int32_t>();
        if (opt.cache != nullptr) {
            if (!kv_cache_append_dev(*opt.cache, il, d_len.as<int32_t>(), pj.kv_cmpr, pj.k_pe, nt, err,
                                     stream))
                return false;
            probe("kv_cache_append_dev");
            // No sync: the append and the attention that reads it are on the same stream, in that order.
            keys = opt.cache->kv_cmpr[(std::size_t) il];
            keys_pe = opt.cache->k_pe[(std::size_t) il];
            keys_format = kv_format(*opt.cache);
            kv_pos = d_kv_pos.as<int32_t>();
        }
        // Update the sparse index even before selection becomes necessary.
        const int32_t* sel = nullptr;
        int n_sel = 0;
        if (!sliding && ix_k.p != nullptr && opt.cache != nullptr && wk.idx != nullptr) {
            const int ilayer = wk.idx->ids[(std::size_t) il];
            probe("before the indexer");
            const IndexerPlan plan = indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, base_pos, nt);
            if (ilayer >= 0) {
                S ix_sc = sc_pool.get(Scratch::kIndexerSc,
                                      indexer_scratch_bytes(g.idx_heads, g.idx_key_dim, g.idx_kpool, nt,
                                                            wk.idx->max_pool, plan.n_sel));
                S ix_sel = sc_pool.get(Scratch::kIndexerSel,
                                       (std::size_t) (window_decode ? std::max(1,max_sel) : (plan.n_sel > 0 ? plan.n_sel : 1)) * nt * 4);
                IndexerWeights iw{};
                iw.attn_k = ix_k.p;            iw.attn_k_type = ix_k.type;
                iw.attn_q_b = ix_qb.p;         iw.attn_q_b_type = ix_qb.type;
                iw.proj = ix_pr.p;             iw.proj_type = ix_pr.type;
                iw.value = ix_cv.p;            iw.value_type = ix_cv.type;
                iw.gate = ix_cg.p;             iw.gate_type = ix_cg.type;
                iw.k_norm = ix_kn.as<float>();
                iw.compressor_norm = ix_cn.as<float>();
                iw.ape = ix_ape.as<float>();
                indexer_select(*wk.idx, ilayer, iw, plan, cur_in.q8, reduced.as<float>(), pj.q_resid_q8,
                               pj.q_resid, mc, n_embd, q_lora, g.idx_heads, n_rot, nt, base_pos,
                               d_pos.as<int32_t>(), rope_base, rms_eps, ix_sel.as<int32_t>(), ix_sc.p,
                               stream, opt.picture ? opt.picture->rope : ImageRope{},
                               opt.dfly != nullptr && opt.dfly->mode == DflyCapture::Mode::verify_window
                                   ? (float*) opt.dfly->vg[(std::size_t) ilayer] : nullptr);
                if (opt.dfly != nullptr && plan.engaged) {
                    // The selection buffer is shared by every sparse block and the next one overwrites it,
                    // so a block the drafter reads is copied out NOW, same stream, before that.
                    DflyCapture& d = *opt.dfly;
                    const bool tail = d.mode == DflyCapture::Mode::prefill_tail;
                    for (std::size_t i = 0; i < d.sel_blocks.size(); ++i) {
                        if (d.sel_blocks[i] != il) continue;
                        if (plan.n_sel > d.sel_cap) { err = "dfly: selection wider than the capture"; return false; }
                        const int rows = tail ? 1 : nt;
                        const int32_t* from = ix_sel.as<int32_t>() + (tail ? (std::size_t) (nt - 1) * plan.n_sel : 0);
                        if (cudaMemcpyAsync(d.sel[i], from, (std::size_t) rows * plan.n_sel * 4,
                                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                            err = "dfly: the selection copy failed at block " + std::to_string(il);
                            return false;
                        }
                    }
                }
                if (plan.engaged) {
                    sel = ix_sel.as<int32_t>();
                    n_sel = plan.n_sel;
                    // Report the first use of sparse selection.
                    static bool said = false;
                    if (!said) {
                        said = true;
                        std::fprintf(stderr,
                                     "hy4_flash indexer: ENGAGED at position %d - %d of %d visible pools kept, "
                                     "%d cells a token\n",
                                     base_pos, plan.n_top, (base_pos + nt) / g.idx_kpool, plan.n_sel);
                    }
                }
            }
        }
        probe("the indexer");
        if (pq_merged) {
            // Every query's own count comes from pq_meta; n_kv is the largest of them.
            mla_attention(absorbed.as<float>(), pj.q_pe, keys, keys_pe, keys_format, sinks.as<float>(), vb.p,
                          vb.type, kv_lora, n_rot, v_head, n_head, nt, base_pos + nt, d_pos.as<int32_t>(), kv_pos,
                          window, 1.0f / std::sqrt((float) qk_head), kqv.as<float>(), at_sc.p, stream, kv_stride,
                          nullptr, sel, n_sel, max_sel, &pq_meta);
        } else if (per_query) {
            // Query j as the decode step at base_pos + j: the host n_kv is what that step's device count
            // holds (len + 1), so passing it with no device count gives the kernel the same partition.
            for (int j = 0; j < nt; ++j)
                mla_attention(absorbed.as<float>() + (std::size_t) j * kv_lora * n_head,
                              pj.q_pe + (std::size_t) j * n_rot * n_head, keys, keys_pe, keys_format,
                              sinks.as<float>(), vb.p, vb.type, kv_lora, n_rot, v_head, n_head, 1,
                              base_pos + j + 1, d_pos.as<int32_t>() + j, kv_pos, window,
                              1.0f / std::sqrt((float) qk_head),
                              kqv.as<float>() + (std::size_t) j * v_head * n_head, at_sc.p, stream, kv_stride,
                              nullptr, sel != nullptr ? sel + (std::size_t) j * n_sel : nullptr, n_sel, max_sel);
        } else {
        mla_attention(absorbed.as<float>(), pj.q_pe, keys, keys_pe, keys_format, sinks.as<float>(), vb.p, vb.type, kv_lora,
                      n_rot, v_head, n_head, nt, n_kv, d_pos.as<int32_t>(), kv_pos, window,
                      1.0f / std::sqrt((float) qk_head), kqv.as<float>(), at_sc.p, stream, kv_stride,
                      opt.cache != nullptr ? d_len.as<int32_t>() + 1 : nullptr, sel, n_sel, max_sel);
        }
        probe("mla_attention");
        // Optionally compare per-query intermediate attention values against a reference.
        static const bool pq_check = [] {
            const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_PQ_CHECK");
            return v != nullptr && std::atoi(v) != 0;
        }();
        if (pq_merged && pq_check) {
            const std::size_t na = (std::size_t) kv_lora * nt * n_head, nk = (std::size_t) v_head * n_head * nt;
            Buf ra(na * 4), rk(nk * 4);
            for (int j = 0; j < nt; ++j) {
                mla_absorb(pj.q + (std::size_t) j * qk_head * n_head, kb.p, kb.type, qk_head, nope, kv_lora,
                           n_head, 1, ra.as<float>() + (std::size_t) j * kv_lora * n_head, a_sc.p, stream);
                mla_attention(ra.as<float>() + (std::size_t) j * kv_lora * n_head,
                              pj.q_pe + (std::size_t) j * n_rot * n_head, keys, keys_pe, keys_format,
                              sinks.as<float>(), vb.p, vb.type, kv_lora, n_rot, v_head, n_head, 1,
                              base_pos + j + 1, d_pos.as<int32_t>() + j, kv_pos, window,
                              1.0f / std::sqrt((float) qk_head),
                              rk.as<float>() + (std::size_t) j * v_head * n_head, at_sc.p, stream, kv_stride,
                              nullptr, sel != nullptr ? sel + (std::size_t) j * n_sel : nullptr, n_sel, max_sel);
            }
            std::vector<uint32_t> ga(na), gk(nk), wa(na), wkq(nk);
            cudaStreamSynchronize(stream);
            cudaMemcpy(ga.data(), absorbed.p, na * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gk.data(), kqv.p, nk * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(wa.data(), ra.p, na * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(wkq.data(), rk.p, nk * 4, cudaMemcpyDeviceToHost);
            // merged absorbed is [kv_lora, nt, n_head]; the reference's query t is [kv_lora, n_head]
            std::size_t bad_a = 0, bad_k = 0;
            for (int h = 0; h < n_head; ++h)
                for (int t = 0; t < nt; ++t)
                    for (int d = 0; d < kv_lora; ++d)
                        bad_a += ga[((std::size_t) h * nt + t) * kv_lora + d] !=
                                 wa[((std::size_t) t * n_head + h) * kv_lora + d];
            for (std::size_t x = 0; x < nk; ++x) bad_k += gk[x] != wkq[x];
            static std::size_t blocks = 0, values = 0, bad_blocks = 0;
            ++blocks;
            values += na + nk;
            if (bad_a != 0 || bad_k != 0) {
                ++bad_blocks;
                std::fprintf(stderr, "DFLY_PQ_CHECK MISMATCH block %d pos %d nt %d: absorbed %zu/%zu, kqv %zu/%zu\n",
                             il, base_pos, nt, bad_a, na, bad_k, nk);
            }
            if (il == n_blocks - 1)
                std::fprintf(stderr, "DFLY_PQ_CHECK pos %d nt %d: %zu blocks, %zu values, %zu blocks differ\n",
                             base_pos, nt, blocks, values, bad_blocks);
        }
        // The attention's own output, BEFORE `mla_output` - which gates `kqv` in place, so after it the raw
        // result is gone. A block's two halves, told apart.
        if (opt.on_attn != nullptr) {
            cudaStreamSynchronize(stream);
            opt.on_attn(il, kqv.as<float>(), opt.user);
        }
        mla_output(kqv.as<float>(), reduced.as<float>(), agate.p, agate.type, wo.p, wo.type, pan.as<float>(), n_embd,
                   v_head * n_head, nt, rms_eps, attn_out.as<float>(), o_sc.p, stream, cur_in, gate_join);
        probe("mla_output");
        // The residual is the state the block RECEIVED - after Engram, where there is one.
        hc_post(attn_out.as<float>(), state.as<float>(), post.as<float>(), n_embd, hc, nt, next.as<float>(),
                stream);
        probe("hc_post (end of the attention half)");
        }  // !replay
        // Partial residency: only the pure-GPU attention half is captured. Host expert planning and
        // every FFN operation still execute below on every token, including a graph replay.
        if (capture_probe && capturing) probe("everything before EndCapture");
        if (partial_attn_graph) {
            if (capturing) {
                cudaGraph_t g{};
                if (cudaStreamEndCapture(stream, &g) != cudaSuccess ||
                    cudaGraphInstantiate(&graph, g, nullptr, nullptr, 0) != cudaSuccess) {
                    err = "forward_prefill: partial attention graph capture failed at block " + std::to_string(il);
                    return false;
                }
                std::size_t nodes = 0;
                cudaGraphGetNodes(g, nullptr, &nodes);
                if(async_decode){std::size_t nodes=0;cudaGraphGetNodes(g,nullptr,&nodes);std::fprintf(stderr,"ASYNC_GRAPH layer=%d nodes=%zu\n",il,nodes); }
            cudaGraphDestroy(g);
                wk.attn_graph[(std::size_t) il] = graph;
                std::fprintf(stderr, "CAUSAL_PARTIAL_GRAPH layer=%d nodes=%zu captured\n", il, nodes);
            }
            if (cudaGraphLaunch(graph, stream) != cudaSuccess) {
                err = "forward_prefill: partial attention graph launch failed at block " + std::to_string(il);
                return false;
            }
        }
        // Swap residual buffers on the host after queuing the current layer.
        std::swap(state, next);
        if (pt != nullptr) {
            cudaStreamSynchronize(stream);
            tick(pt->attention, t_at);
        }
        // Freed before the FFN half uploads its own, so a pass that streams peaks at one half of one
        // block. With a residency these are no-ops: the whole point is that the next step finds them.
        hcfn.reset(); qa.reset(); qb.reset(); kvm.reset(); kb.reset(); vb.reset(); agate.reset(); wo.reset();

        // Capture the FFN with attention when expert execution supports it.
        const auto t_fw = std::chrono::steady_clock::now();
        if (!replay || partial_attn_graph) {
        W hcffn = take(gg, rs, budget, blk(il) + "hc_ffn_fn.weight");
        W hcfs = take(gg, rs, budget, blk(il) + "hc_ffn_scale.weight");
        W hcfb = take(gg, rs, budget, blk(il) + "hc_ffn_base.weight");
        W fnorm = take(gg, rs, budget, blk(il) + "ffn_norm.weight");
        W pfn = take(gg, rs, budget, blk(il) + "post_ffw_norm.weight");
        hc_pre(state.as<float>(), hcffn.p, hcfs.as<float>(), hcfb.as<float>(), n_embd, hc, nt, 2 * hc,
               rms_eps, hc_eps, magnitude, scratch_hc.p, mixes.as<float>(), pre.as<float>(),
               post.as<float>(), reduced.as<float>(), stream, mc, hcffn.type);
        // And the same for this half: the router and the shared expert read one normed hidden and one Q8.
        S f_cur_q8 = sc_pool.get(Scratch::kCurQ8, kernels::native_q8_1_bytes_wide(n_embd, nt));
        if (!kernels::native_rms_norm_quantize_q8_1(reduced.as<float>(), fnorm.as<float>(),
                                                    reduced.as<float>(), f_cur_q8.p, nt, n_embd, rms_eps,
                                                    stream)) {
            kernels::rms_norm_weighted_wide(reduced.as<float>(), fnorm.as<float>(), nt, n_embd, rms_eps, stream);
            kernels::native_quantize_q8_1_wide(reduced.as<float>(), f_cur_q8.p, n_embd, nt, stream);
        }
        // The pair this half's projections read `reduced` through - the router, the shared expert and the
        // experts' down-projection all take it.
        const MatIn f_cur_in = mat_staged(mc, reduced.as<float>(), f_cur_q8.p, n_embd, nt, stream);
        if (pt != nullptr) tick(pt->weights, t_fw);

        const auto t_ff = std::chrono::steady_clock::now();
        if (spec.expert_ordinal < 0) {
            W fg = take(gg, rs, budget, blk(il) + "ffn_gate.weight");
            W fu = take(gg, rs, budget, blk(il) + "ffn_up.weight");
            W fd = take(gg, rs, budget, blk(il) + "ffn_down.weight");
            const int n_ff = (int) fg.dim(1);
            S f_sc = sc_pool.get(Scratch::kFfnSc, dense_ffn_scratch_bytes(n_embd, n_ff, nt));
            if (!fg.p || !fu.p || !fd.p || !f_sc.p) {
                err = "forward_prefill: block " + std::to_string(il) + "'s dense FFN weights";
                return false;
            }
            // UNCLAMPED: this architecture never reads swiglu_clamp_shexp, so only the routed experts are
            // clamped.
            dense_ffn(reduced.as<float>(), fg.p, fg.type, fu.p, fu.type, fd.p, fd.type, pfn.as<float>(), n_embd, n_ff,
                      nt, rms_eps, ffn_out.as<float>(), f_sc.p, stream, f_cur_in);
        } else {
            int t_ge = 0, t_ue = 0, t_de = 0;
            std::vector<int64_t> ge_shape;
            W gi = take(gg, rs, budget, blk(il) + "ffn_gate_inp.weight");
            W pb = take(gg, rs, budget, blk(il) + "exp_probs_b.bias");
            W rd = take(gg, rs, budget, blk(il) + "ffn_routed_down.weight");
            W ru = take(gg, rs, budget, blk(il) + "ffn_routed_up.weight");
            // NOT uploaded: the router picks 16 of 512, so the block's 2.35 GB of experts is fetched a
            // slice at a time below. Their types and shapes still come from the file's own directory.
            const TensorInfo* ti_ge = gg.info(blk(il) + "ffn_gate_exps.weight");
            const TensorInfo* ti_ue = gg.info(blk(il) + "ffn_up_exps.weight");
            const TensorInfo* ti_de = gg.info(blk(il) + "ffn_down_exps.weight");
            if (ti_ge == nullptr || ti_ue == nullptr || ti_de == nullptr) {
                err = "forward_prefill: block " + std::to_string(il) + "'s expert tensors";
                return false;
            }
            t_ge = (int) ti_ge->type;
            t_ue = (int) ti_ue->type;
            t_de = (int) ti_de->type;
            ge_shape.clear();
            for (uint64_t d : ti_ge->shape) ge_shape.push_back((int64_t) d);
            W gs = take(gg, rs, budget, blk(il) + "ffn_gate_shexp.weight");
            W us = take(gg, rs, budget, blk(il) + "ffn_up_shexp.weight");
            W ds = take(gg, rs, budget, blk(il) + "ffn_down_shexp.weight");
            if (!gi.p || !rd.p || !ru.p || !gs.p || !us.p || !ds.p) {
                err = "forward_prefill: block " + std::to_string(il) + "'s MoE weights";
                return false;
            }
            const int expert_dim = (int) rd.dim(1), expert_ff = (int) ge_shape[0];
            const int n_expert = (int) gi.dim(1);
            static const bool resident_mmq = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_RESIDENT_MMQ");
                return v != nullptr && std::atoi(v) != 0;
            }();
            // Only the slot-indexed route tables grow, roughly 12 bytes per pool slot. Without this
            // reserve the resident pass always declines a pool larger than one layer and restages hits.
            const int mmq_slots = resident_mmq && nt > 1 && !slices.complete && slices.pool_i >= 0
                                  ? std::max(n_expert, slices.pools[slices.pool_i].n) : n_expert;
            const int mmq_rows = !slices.complete && (!g_layer_major || g_layer_major->stream_experts)
                                 ? moe_group_rows(g.top_k, nt) : 0;
            // Experimental exact expert deduplication. H20 measurements did not beat the token loop,
            // so it requires an explicit switch even inside the opt-in Dfly path.
            static const bool moe_window_env = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_MOE_WINDOW");
                const char* loop = std::getenv("STRATA_HY4_FLASH_DFLY_MOE_LOOP");
                return v && std::atoi(v) != 0 && !(loop && std::atoi(loop) != 0);
            }();
            // Batch all token/expert pairs without changing the per-token arithmetic.
            static const bool moe_wide_env = [] {
                const char* v = std::getenv("STRATA_HY4_FLASH_DFLY_MOE_WIDE");
                const char* loop = std::getenv("STRATA_HY4_FLASH_DFLY_MOE_LOOP");
                return v && std::atoi(v) != 0 && !(loop && std::atoi(loop) != 0);
            }();
            const bool moe_window = opt.exact_columns && nt > 1 && moe_window_env;
            const bool moe_wide = opt.exact_columns && nt > 1 && moe_wide_env && !moe_window;
            S m_sc = sc_pool.get(Scratch::kFfnSc,
                                 latent_moe_scratch_bytes(n_embd, expert_dim, expert_ff, n_expert,
                                                          g.top_k, nt, t_ge, t_ue, t_de, mmq_slots, mmq_rows,
                                                          moe_window || moe_wide));
            if (!m_sc.p) {
                err = "forward_prefill: block " + std::to_string(il) + "'s MoE scratch";
                return false;
            }
            LatentMoeWeights mw{};
            // The resident stripes for every verify window, MTP's and Dfly's: the same per-entry arithmetic
            // as the token loop, one launch a stage. An explicit DFLY_MOE_WINDOW/WIDE request keeps the loop
            // it asks for; STRATA_HY4_FLASH_RESIDENT_STRIPES=0 turns the stripes off for both.
            mw.verify_exact = (opt.decode_equivalent || (opt.exact_columns && !moe_window && !moe_wide)) && nt <= 8;
            mw.context_n_tok = opt.exact_columns ? 1 : 0;
            mw.exact_window = moe_window;
            mw.exact_wide = moe_wide;
            mw.stage_scratch_slots = mmq_slots;
            mw.stage_scratch_rows = mmq_rows;
            mw.gate_inp = gi.as<float>();
            mw.exp_probs_b = pb.p != nullptr ? pb.as<float>() : nullptr;
            mw.routed_down = rd.p;  mw.routed_down_type = rd.type;
            mw.gate_exps = nullptr; mw.gate_exps_type = t_ge;
            mw.up_exps = nullptr;   mw.up_exps_type = t_ue;
            mw.down_exps = nullptr; mw.down_exps_type = t_de;
            if (!window_decode && !async_decode && !slices.open_block(gg, blk(il), il, expert_dim, expert_ff, stream)) {
                err = "forward_prefill: cannot open block " + std::to_string(il) + "'s expert slices";
                return false;
            }
            if (slices.n_slots <= 0) {
                err = "forward_prefill: block " + std::to_string(il) + " has nowhere to put its experts";
                return false;
            }
            // A full tier maps expert IDs to slots on device.
            ExpertGroup rg{};
            // The resident staging plan must outlive latent_moe.
            ExpertGroup rg_part{};
            if (g_layer_major && !slices.complete && !wk.borrowed_layer && !g_layer_major->stream_experts) {
                auto& lm = *g_layer_major;
                const size_t bytes = tensor_payload_bytes(*ti_ge) + tensor_payload_bytes(*ti_ue)
                                   + tensor_payload_bytes(*ti_de);
                if (lm.weight_capacity < bytes) {
                    lm.weights.reset();
                    // Every plane is populated below before use. On small cards keep the layer
                    // traversal and let the regular expert fetcher stage smaller batches.
                    const auto status = cudaMalloc(&lm.weights.p, bytes);
                    if (status == cudaErrorMemoryAllocation) {
                        cudaGetLastError();
                        lm.stream_experts = true;
                        lm.weight_capacity = 0;
                    } else if (status != cudaSuccess) {
                        err = std::string("layer-major: expert allocation failed: ") + cudaGetErrorString(status);
                        return false;
                    } else {
                        lm.weight_capacity = bytes;
                    }
                }
            }
            if(window_decode) {
                mw.window=&slices.window_exec.layers[il]->device;
            } else if (async_decode) {
                mw.async_group=&async_groups[(std::size_t)il];
                static const bool direct_cpu_parts=[] {
                    const char* e=std::getenv("STRATA_HY4_FLASH_CPU_DIRECT_PARTS");
                    return e && std::atoi(e)!=0;
                }();
                if (direct_cpu_parts && mw.async_group->cpu_parts != nullptr &&
                    (!slices.async_layers[(std::size_t)il] ||
                     slices.async_layers[(std::size_t)il]->cpu_dim != expert_dim ||
                     nt != 1 || g.top_k != 16)) {
                    err="forward_prefill: direct CPU parts allocation shape mismatch";
                    return false;
                }
            } else if (g_layer_major && !g_layer_major->stream_experts && !slices.complete) {
                auto& lm=*g_layer_major;
                if(lm.loaded!=il){
                    const TensorInfo* ts[3]={ti_ge,ti_ue,ti_de};std::size_t per[3],off[3],bytes=0;
                    for(int r=0;r<3;++r){per[r]=tensor_payload_bytes(*ts[r])/n_expert;off[r]=bytes;bytes+=per[r]*n_expert;}
                    if(wk.borrowed_layer){
                        if(bytes>wk.borrowed_layer_bytes){err="layer-major: borrowed layer buffer is too small";return false;}
                        lm.weights=Buf::borrowed(wk.borrowed_layer);lm.weight_capacity=wk.borrowed_layer_bytes;
                    }
                    const auto& pool=slices.pools[slices.pool_i];
                    for(int r=0;r<3;++r)for(int e=0;e<n_expert;){
                        const int slot=slices.slot_of[(std::size_t)il*n_expert+e];int run=1;
                        while(e+run<n_expert){const int next=slices.slot_of[(std::size_t)il*n_expert+e+run];
                            if(slot<0 ? next>=0 : (next!=slot+run || pool.stride_r[r]!=per[r]))break;++run;}
                        if (slices.exclusive && slot < 0) {
                            auto* dst = static_cast<char*>(lm.weights.p) + off[r] + size_t(e)*per[r];
                            for (int j = e; j < e + run; ++j) {
                                const uint64_t from = gg.data_start() + ts[r]->offset + size_t(j)*per[r];
                                if (!slices.read_plane(gg, il*n_expert+j, r, from, slices.exchange, per[r]) ||
                                    cudaMemcpyAsync(dst, slices.exchange, per[r], cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                                    cudaStreamSynchronize(stream) != cudaSuccess) {
                                    err = "layer-major: resident expert staging failed"; return false;
                                }
                                dst += per[r];
                            }
                            lm.h2d += size_t(run)*per[r]; e += run; continue;
                        }
                        const void* src=slot>=0?static_cast<const char*>(slices.tier.p)+pool.plane[r]+(std::size_t)slot*pool.stride_r[r]
                            :gg.at(gg.data_start()+ts[r]->offset+(std::size_t)e*per[r],(std::size_t)run*per[r]);
                        void* dst=static_cast<char*>(lm.weights.p)+off[r]+(std::size_t)e*per[r];
                        const size_t bytes=(std::size_t)run*per[r];
                        if(!src || (slot>=0 ? cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream)
                                            : gg.copy_expert_async(dst,src,bytes,stream))!=cudaSuccess){err="layer-major: expert staging failed";return false;}
                        (slot>=0?lm.d2d:lm.h2d)+=(std::size_t)run*per[r];e+=run;
                    }
                    lm.group.gate=static_cast<char*>(lm.weights.p)+off[0];lm.group.gate_stride=per[0];
                    lm.group.up=static_cast<char*>(lm.weights.p)+off[1];lm.group.up_stride=per[1];
                    lm.group.down=static_cast<char*>(lm.weights.p)+off[2];lm.group.down_stride=per[2];
                    lm.group.index=nullptr;lm.loaded=il;
                    if(lm.identity_count!=n_expert){
                        lm.identity.reset();lm.identity=Buf((std::size_t)n_expert*4);
                        std::vector<int32_t> ident(n_expert);for(int e=0;e<n_expert;++e)ident[e]=e;
                        if(!lm.identity.p || cudaMemcpy(lm.identity.p,ident.data(),ident.size()*4,cudaMemcpyHostToDevice)!=cudaSuccess){err="layer-major: expert identity map failed";return false;}
                        lm.identity_count=n_expert;
                    }
                    // Same free-slot-only prompt admission as B2. The whole layer stays readable until
                    // every chunk has consumed it; decode's tier never aliases this temporary.
                    ExpertSlices::PrefillRing bank;bank.dev=lm.weights.p;
                    for(int r=0;r<3;++r){bank.plane[r]=off[r];bank.stride_r[r]=per[r];}
                    std::vector<int32_t> all(n_expert);for(int e=0;e<n_expert;++e)all[e]=e;
                    const char* admit=std::getenv("STRATA_HY4_FLASH_PREFILL_ADMIT");
                    if(!admit || std::atoi(admit)!=0){
                        slices.prefill_ring_admit(bank,il,all,stream);
                        if(wk.borrow_phase==1)for(int e=0;e<n_expert;++e)
                            wk.borrowed_observed[(std::size_t)il*n_expert+e]=1;
                    }
                }
                mw.resident=&lm.group;mw.resident_base=0;mw.resident_slot_of=lm.identity.as<int32_t>();
            } else if (slices.complete && slices.resident_group(il, &rg)) {
                mw.resident = &rg;
                mw.resident_slot_of = static_cast<const int32_t*>(slices.slot_dev.p) +
                                      (std::size_t) il * (std::size_t) n_expert;
                mw.resident_base = slices.block_base[(std::size_t) il];
            } else {
                mw.fetch_experts = fetch_experts_cb;
                mw.fetch_user = &slices;
                // Batch prompt experts in a staging ring. More slots reduce the number of MMQ rounds
                // per layer at the cost of additional device memory.
                static const int ring_slots = [] {
                    const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_OFFLOAD_MMQ");
                    const int n = v != nullptr ? std::atoi(v) : 0;
                    return n > 0 ? n : 0;
                }();
                if (ring_slots > 0 && nt > 1) {
                    std::string rerr;
                    if (slices.prefill_ring_open(ring_slots, n_expert, rerr)) {
                        mw.stage_experts = stage_experts_cb;
                        if (slices.prefill_pipeline_on()) {
                            mw.stage_plan = stage_plan_cb;
                            mw.stage_release = stage_release_cb;
                        }
                        mw.stage_user = &slices;
                        mw.stage_batch_max = slices.pring.slots;
                        mw.stage_n_slot = slices.pring.slots;
                        mw.stage_block = il;
                        // Read resident weights directly using the tier's slot map.
                        const bool rg_ok = slices.pool_i >= 0 && slices.resident_group_raw(il, &rg_part);
                        const int32_t* ts = rg_ok ? slices.prefill_tier_slots(il, stream) : nullptr;
                        {
                            static int dsaid = 0;
                            if (dsaid++ == 0)
                                std::fprintf(stderr, "HY4_FLASH_PREFILL_RESIDENT_WIRE pool_i=%d rg=%d ts=%d\n",
                                             slices.pool_i, rg_ok ? 1 : 0, ts != nullptr ? 1 : 0);
                        }
                        if (rg_ok) {
                            if (ts != nullptr) {
                                mw.stage_slot_of = ts;
                                mw.stage_resident = &rg_part;
                                const int base = slices.pring.tier_slot_base;
                                rg_part.gate = static_cast<const char*>(rg_part.gate) + (std::size_t) base * rg_part.gate_stride;
                                rg_part.up = static_cast<const char*>(rg_part.up) + (std::size_t) base * rg_part.up_stride;
                                rg_part.down = static_cast<const char*>(rg_part.down) + (std::size_t) base * rg_part.down_stride;
                                mw.stage_resident_slots = slices.pring.tier_slot_count;
                            }
                        }
                    } else {
                        static int said = 0;
                        if (said++ == 0)
                            std::fprintf(stderr, "hy4_flash: the prompt expert ring is off: %s\n",
                                         rerr.c_str());
                    }
                }
            }
            mw.routed_up = ru.p;    mw.routed_up_type = ru.type;
            mw.gate_shexp = gs.p;   mw.gate_shexp_type = gs.type;
            mw.up_shexp = us.p;     mw.up_shexp_type = us.type;
            mw.down_shexp = ds.p;   mw.down_shexp_type = ds.type;
            mw.post_norm = pfn.as<float>();
            LatentMoeTrace tr{};
            latent_moe(reduced.as<float>(), mw, n_embd, expert_dim, expert_ff, n_expert, g.top_k, nt,
                       g.routed_scale, g.swiglu_clamp, rms_eps, ffn_out.as<float>(), m_sc.p, stream,
                       // Collect routing traces only when requested to avoid host-visible work on the normal
                       // path.
                       opt.on_routing != nullptr ? &tr : nullptr,
                       // AND THE MatIn, NOT THE RAW Q8 (ours): the projections dispatch on it.
                       f_cur_in);
            if (opt.on_routing != nullptr) {
                cudaStreamSynchronize(stream);
                opt.on_routing(il, tr.ids, opt.user);
            }
        }
        if (pt != nullptr) {
            cudaStreamSynchronize(stream);
            tick(spec.expert_ordinal < 0 ? pt->dense_ffn : pt->moe, t_ff);
        }
        probe("the MoE half (latent_moe)");
        hc_post(ffn_out.as<float>(), state.as<float>(), post.as<float>(), n_embd, hc, nt, next.as<float>(),
                stream);
        probe("the MoE hc_post - the whole block");
        }  // !replay
        if (capturing && !partial_attn_graph) {
            cudaGraph_t g{};
            if (cudaStreamEndCapture(stream, &g) != cudaSuccess ||
                cudaGraphInstantiate(&graph, g, nullptr, nullptr, 0) != cudaSuccess) {
                err = "forward_prefill: graph capture failed at block " + std::to_string(il);
                return false;
            }
            if(async_decode){std::size_t nodes=0;cudaGraphGetNodes(g,nullptr,&nodes);std::fprintf(stderr,"ASYNC_GRAPH layer=%d nodes=%zu\n",il,nodes); }
            // Report each verification width when its implementation actually runs.
            if (spec_capture && il == 0) {
                std::size_t nodes = 0;
                cudaGraphGetNodes(g, nullptr, &nodes);
                std::fprintf(stderr, "HY4_FLASH_VERIFY_GRAPH captured nt=%d blocks=%d nodes_block0=%zu ffn=%d\n",
                             nt, n_blocks, nodes, partial_attn_graph ? 0 : 1);
            }
            cudaGraphDestroy(g);
            wk.attn_graph[(std::size_t) il] = graph;
        }
        // Capture RECORDS rather than runs, so the launch is what executes this block - on the pass that
        // captured it as much as on every pass after.
        if (can_capture && !partial_attn_graph && cudaGraphLaunch(graph, stream) != cudaSuccess) {
            err = "forward_prefill: graph launch failed at block " + std::to_string(il);
            return false;
        }
        std::swap(state, next);

        if(window_expected && !token_capturing && !slices.wait_window_layer(il,window_expected)) {err="window layer supply failed";return false;}
        if(async_this_layer) {
            async_last_layer=il;async_last_expected=async_expected;
            // Eager launch backpressure and first-capture CUDA allocation remain bounded. Warm graph
            // replay submits all blocks without per-layer CPU completion waits.
            if(!token_capturing && (!can_capture || capturing) && !slices.wait_async_layer(il,async_expected)){
                err="async layer completion failed at "+std::to_string(il);return false;
            }
        }
        // Wait for device work before invoking a host observer.
        const bool host_reads_state = opt.on_block != nullptr || opt.teacher_state != nullptr;
        if (host_reads_state && cudaStreamSynchronize(stream) != cudaSuccess) {
            err = "forward_prefill: block " + std::to_string(il) + " did not complete";
            return false;
        }
        if (opt.on_block != nullptr && !opt.on_block(il, state.as<float>(), opt.user)) {
            err = "forward_prefill: the caller stopped the pass at block " + std::to_string(il);
            return false;
        }
        if (opt.teacher_state != nullptr)
            if (const float* t = opt.teacher_state(il, opt.user)) {
                cudaMemcpy(state.p, t, (std::size_t) n_embd * hc * nt * 4, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(stream);
            }
    }

    // Advance the shared pool counters after every sparse layer has consumed the chunk.
    if (wk.idx != nullptr && opt.cache != nullptr && g.idx_kpool > 0)
        indexer_advance(*wk.idx, nt, stream);

    // Drain the last asynchronous layer before the head consumes its output.
    if (!token_capturing && async_last_layer >= 0 && !slices.wait_async_layer(async_last_layer, async_last_expected)) {
        err = "forward_prefill: the async expert service did not complete its last layer";
        return false;
    }
    if (!token_capturing && !slices.drain_async()) {
        err = "forward_prefill: the async expert service could not be drained";
        return false;
    }
    // ---- the output head, unless this is a prompt chunk nothing reads (opt.want_logits)
    t_hd = std::chrono::steady_clock::now();
    if (opt.want_logits || opt.hidden_out) {
    if (logit_rows) {
        const auto bytes = (std::size_t)vocab * logit_rows * sizeof(float);
        if (opt.device_logits_out && wk.borrow_phase == 1) {
            wk.prefill_device_logits = Buf(bytes);
            if (!wk.prefill_device_logits.p) { err="prefill loan: device logits allocation failed"; return false; }
            d_logits = {wk.prefill_device_logits.p};
        } else {
            d_logits = sc_pool.get(Scratch::kLogits, bytes);
        }
    }
    {
        W hfn = take(gg, head_slot, budget, "output_hc_fn.weight");
        W hcs = take(gg, head_slot, budget, "output_hc_scale.weight");
        W hcb = take(gg, head_slot, budget, "output_hc_base.weight");
        W on = take(gg, head_slot, budget, "output_norm.weight");
        W ow = take(gg, head_slot, budget, "output.weight");
        S sc = sc_pool.get(Scratch::kHeadSc, output_head_scratch_bytes(n_embd, hc, vocab, head_rows));
        if (!hfn.p || !ow.p || !sc.p) {
            // Include the failing allocation in the error report.
            std::size_t mfree = 0, mtot = 0;
            cudaMemGetInfo(&mfree, &mtot);
            cudaGetLastError();
            const double mb = 1.0 / (1024.0 * 1024.0);
            err = "forward_prefill: the output head did not fit beside a " + std::to_string(nt) +
                  "-token chunk's buffers (" + std::string(hfn.p == nullptr ? "output_hc_fn " : "") +
                  std::string(ow.p == nullptr ? "output.weight " : "") +
                  std::string(sc.p == nullptr ? "head scratch " : "") + "failed; " +
                  std::to_string((long) (mfree * mb)) + " MiB free of " +
                  std::to_string((long) (mtot * mb)) + ", head scratch wants " +
                  std::to_string((long) (output_head_scratch_bytes(n_embd, hc, vocab, head_rows) * mb)) +
                  " MiB for " + std::to_string(head_rows) + " rows) - lower STRATA_HY4_FLASH_PREFILL_CHUNK or "
                  "--expert-vram";
            return false;
        }
        // The LAST row only, when that is all the caller reads: `state` is [n_embd, hc, n_tok] with token
        // t at `t * n_embd * hc`, so the final token's slice is a pointer offset and nothing else changes.
        output_head(state.as<float>() + (std::size_t) (nt - head_rows) * n_embd * hc, hfn.p, hfn.type,
                    hcs.as<float>(), hcb.as<float>(), on.as<float>(), ow.p, ow.type, n_embd, hc, vocab,
                    head_rows, rms_eps, hc_eps, logit_rows ? d_logits.as<float>() : nullptr, sc.p,
                    stream, mc, window_graph_path?static_cast<float*>(wk.window_hidden.p):opt.hidden_out, logit_rows,
                    opt.dfly != nullptr ? (float*) opt.dfly->hidden : nullptr,
                    opt.dfly == nullptr ? 0
                        : (opt.dfly->mode == DflyCapture::Mode::prefill_tail ? 1 : nt));
        if (!token_capturing) cudaStreamSynchronize(stream);
    }
    }
    } // !token_replay: recording includes the original head and original shared-stream joins

    // Keep the host counter mirror consistent with the device update.
    if (wk.idx != nullptr && opt.cache != nullptr && g.idx_kpool > 0) {
        const int have = wk.idx->carry + nt;
        wk.idx->n_pool += have / g.idx_kpool;
        wk.idx->carry = have % g.idx_kpool;
        wk.idx->valid_pos = base_pos + nt;
    }
    if (token_capturing) {
        cudaGraph_t graph = nullptr;
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        const auto capture_end = std::chrono::steady_clock::now();
        token_capture_guard.active = false;
        if (ended != cudaSuccess || !graph) {
            if (graph) cudaGraphDestroy(graph);
            err = "forward_prefill: token graph end capture failed";
            return false;
        }
        std::size_t nodes = 0;
        cudaGraphGetNodes(graph, nullptr, &nodes);
        cudaGraphExec_t* destination = verify_capturing ? &verify_entry->exec : &pass_graph;
        const auto instantiate_start = std::chrono::steady_clock::now();
        const cudaError_t instantiated = cudaGraphInstantiate(destination, graph, nullptr, nullptr, 0);
        const auto instantiate_end = std::chrono::steady_clock::now();
        const char* loop_mode=std::getenv("STRATA_HY4_FLASH_DEVICE_LOOP");
        if(window_graph_path&&loop_mode&&std::atoi(loop_mode))wg.source=graph;
        else cudaGraphDestroy(graph);
        if (instantiated != cudaSuccess) {
            if (verify_capturing) wk.invalidate_verify_graphs();
            else if(window_graph_path)wk.invalidate_window_graphs();else wk.invalidate_token_graph();
            err = "forward_prefill: token graph instantiate failed";
            return false;
        }
        if (verify_capturing)
        std::fprintf(stderr, "%s captured T=%d carry=%d nodes=%zu scratch_generation=%llu capture_ms=%.3f instantiate_ms=%.3f\n",
                     window_graph_path?"HY4_FLASH_WINDOW_GRAPH":"HY4_FLASH_TOKEN_GRAPH",nt,base_pos%4,nodes,(unsigned long long)sc_pool.generation,
                     std::chrono::duration<double, std::milli>(capture_end-capture_start).count(),
                     std::chrono::duration<double, std::milli>(instantiate_end-instantiate_start).count());
    }
    if (token_capturing || token_replay) {
        if (cudaGraphLaunch(verify_entry ? verify_entry->exec : pass_graph, stream) != cudaSuccess) {
            err = "forward_prefill: token graph launch failed";
            return false;
        }
        if (verify_replay && !verify_entry->replay_reported) {
            std::fprintf(stderr, "HY4_FLASH_DFLY_GRAPH replay w=%d engaged=%d position=%d\n", nt,
                         (int)idx_engaged, base_pos);
            verify_entry->replay_reported = true;
        }
        if(window_graph_path) {
            if(window_last<0 || !slices.wait_window_layer(window_last,window_epoch) || !slices.drain_window()) {
                slices.window_fail("graph completion");cudaStreamSynchronize(stream);
                err="window graph expert service failed";return false;
            }
        } else if (token_path && (async_last_layer < 0 || !slices.wait_async_layer(async_last_layer, async_last_expected) ||
            !slices.drain_async() || slices.async_error())) {
            // Release every layer's wait even if completion itself timed out, then finish all
            // already submitted readers before caller teardown or a possible retry. Destroying a
            // CUDA stream alone is asynchronous and is not a lifetime barrier.
            if (!slices.async_error()) slices.async_fail(8, "token graph completion failed");
            cudaStreamSynchronize(stream);
            err = "forward_prefill: token graph expert service failed; output not committed";
            logits_out.clear();
            return false;
        }
    }
    if(window_graph_path && opt.hidden_out &&
       cudaMemcpyAsync(opt.hidden_out,wk.window_hidden.p,(std::size_t)n_embd*head_rows*4,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess) {
        err="window hidden result copy failed";return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        err = "forward_prefill: the pass did not complete on the device";
        return false;
    }
    if(window_decode && !slices.drain_window()){err="window supply failed; output not committed";logits_out.clear();return false;}
    window_guard.complete=true;
    if(g_layer_major){
        auto& lm=*g_layer_major;
        // Copy the FINAL state, not the initial kState pointer: Engram adds one ping-pong swap.
        if(lm.state_device){
            if(cudaMemcpyAsync(lm.state,state.p,(std::size_t)n_embd*hc*nt*4,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess ||
               cudaStreamSynchronize(stream)!=cudaSuccess){err="layer-major: device state store failed";return false;}
        } else if(cudaMemcpy(lm.state,state.p,(std::size_t)n_embd*hc*nt*4,cudaMemcpyDeviceToHost)!=cudaSuccess){
            err="layer-major: state download failed";return false;}
    }
    if (opt.want_logits) {
        if(opt.device_logits_out){*opt.device_logits_out=d_logits.as<float>();logits_out.clear();}
        else {
            logits_out.resize((std::size_t) vocab * logit_rows);
            const cudaError_t d2h = cudaMemcpy(logits_out.data(), d_logits.p, logits_out.size() * 4, cudaMemcpyDeviceToHost);
            if (opt.dfly != nullptr && d2h != cudaSuccess) {
                cudaGetLastError();
                err = "dfly: the logits could not be copied back";
                logits_out.clear();
                return false;
            }
        }
        if (opt.dfly != nullptr) {
            // Every copy above is on `stream` and the sync above succeeded, so the capture is complete.
            DflyCapture& d = *opt.dfly;
            const bool tail = d.mode == DflyCapture::Mode::prefill_tail;
            d.has_window = !tail;
            d.pos_begin = tail ? base_pos + nt - 1 : base_pos;
            d.n_rows = tail ? 1 : nt;
            d.engaged = indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, d.pos_begin, tail ? 1 : nt).engaged;
            d.n_sel = d.engaged ? indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, d.pos_begin, tail ? 1 : nt).n_sel : 0;
            d.win_start = base_pos - base_pos % g.idx_kpool;
            d.win_rows = base_pos % g.idx_kpool + nt;
            d.generation += 1;
            d.valid = true;
        }
    } else {
        logits_out.clear();
    }
    if (pt != nullptr) tick(pt->head, t_hd);
    if(!wk.slices.drain_async() || wk.slices.async_error()){err="async MoE supply failed; output not committed";logits_out.clear();return false;}
    if (verify_path && !verify_capturing && !verify_replay) {
        if (wk.verify_scratch_generation != sc_pool.generation) wk.invalidate_verify_graphs();
        wk.verify_scratch_generation = sc_pool.generation;
        verify_entry->warm = true;
    }
    // Unset HY4_FLASH_DEVICE_PLAN_STATS to disable reporting. A value of zero still enables it.
    static const bool plan_stats = std::getenv("HY4_FLASH_DEVICE_PLAN_STATS") != nullptr;
    if (plan_stats) wk.slices.report_device_plan();
    if (token_path && !token_capturing && !token_replay) {
        // This was a real eager token, not a duplicate warmup that could mutate KV/history twice.
        // A newly encountered shape may have grown scratch inside the layer loop.
        // Retire earlier shapes before adopting that generation as the new baseline.
        if(window_graph_path && wk.window_generation!=sc_pool.generation)wk.invalidate_window_graphs();
        pass_warm = true;
        if(window_graph_path)wk.window_generation=sc_pool.generation;
        wk.token_scratch_generation = sc_pool.generation;
        std::fprintf(stderr, "%s eager_warmup complete scratch_generation=%llu\n",
                     window_graph_path?"HY4_FLASH_WINDOW_GRAPH":"HY4_FLASH_TOKEN_GRAPH",(unsigned long long)sc_pool.generation);
    }
    if(!g_layer_major && async_option && nt==1 && base_pos>0)wk.async_primed=true;
    // Mark the verification width warm only after a full window completes.
    if(!g_layer_major && opt.speculative && opt.decode_equivalent && nt>1 && base_pos>0)wk.spec_warm_nt=nt;

    if(allow_borrow && opt.want_logits && wk.borrow_phase==1){
        if(g_layer_major)g_layer_major->weights.reset();
        if(!prefill_phase_transition(res,g,opt,false,1,stream,err)){logits_out.clear();return false;}
    }
    // Advanced only now: a pass that failed half way leaves the cache as it was, so a caller can retry
    // rather than find the sequence half extended.
    if (opt.cache != nullptr) opt.cache->len += nt;
    if (res != nullptr) {
        res->tier_slots = wk.slices.n_slots;
        res->tier_filled = wk.slices.filled();
        res->tier_bytes = wk.slices.tier_bytes();
        res->tier_hits = wk.slices.hits;
        res->tier_misses = wk.slices.misses;
        res->cpu_experts = wk.slices.cpu_jobs.load();
        res->cpu_result_bytes = wk.slices.cpu_result_bytes.load();
        res->h2d_bytes = wk.slices.h2d_bytes;
        res->d2d_bytes = wk.slices.d2d_bytes;
        res->offload={wk.slices.h2d_bytes+wk.slices.prefill_weight_bytes.load(),
            (uint64_t)wk.slices.n_fetch+wk.slices.prefill_weight_transfers.load(),
            (uint64_t)wk.slices.hits,(uint64_t)wk.slices.misses,
            wk.slices.window_exec.windows,wk.slices.window_exec.entries,wk.slices.window_exec.unique};
    }
    if (pt != nullptr) {
        tick(pt->total, t_call);
        ++pt->passes;
        // Subtract the previous cumulative counters to report this pass.
        pt->fetch_wait += slices.t_wait - pt->seen_wait;
        pt->fetch_read += slices.t_read - pt->seen_read;
        pt->fetch_post += slices.t_post - pt->seen_post;
        pt->fetches += slices.n_fetch - pt->seen_fetch;
        pt->seen_wait = slices.t_wait;
        pt->seen_read = slices.t_read;
        pt->seen_post = slices.t_post;
        pt->seen_fetch = slices.n_fetch;
    }
    // Every dense name this geometry uses is now in the residency's map, so no later pass uploads and a
    // capture has nothing it cannot contain. See Work::warm.
    if (res != nullptr && res->work != nullptr) res->work->warm = true;
    guard.completed = true;
    return true;
} catch (const DenseArenaError& e) { err=e.what(); logits_out.clear(); return false; }


bool expert_window_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_HY4_FLASH_WINDOW");
        return value && std::atoi(value) != 0;
    }();
    return enabled;
}

bool dense_experts_complete(const DenseResidency& res) {
    // `Work` and its `ExpertSlices` are private to this file, so the question has to be answered here
    // rather than by the caller reaching in. Null `work` means no pass has run yet.
    return res.work != nullptr && res.work->slices.complete;
}

bool forward_prefill(const std::string& gguf_path, const Geometry& g, const std::string& pack_dir,
                     const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err,
                     const ForwardOptions& opt) {
    struct MatMode {
        bool previous=mat_decode_equivalent();
        ~MatMode(){mat_set_decode_equivalent(previous);}
    } mode;
    mat_set_decode_equivalent(opt.decode_equivalent);
    if(!opt.resident_experts && !auto_chunk_on() && !prefill_borrow_on())
        return forward_prefill_impl(gguf_path,g,pack_dir,tokens,logits_out,err,opt);
    if(opt.dense && opt.dense->work && opt.dense->work->borrow_failed){
        err="previous forward pass failed; recreate the session";logits_out.clear();return false;
    }
    // A failed pass may already have written KV/indexer contents even if its host length was not
    // committed. Do not retry against those contents or publish a partially returned expert arena.
    try {
        if(forward_prefill_impl(gguf_path,g,pack_dir,tokens,logits_out,err,opt)){
            if(opt.want_logits && opt.dense && opt.dense->work)opt.dense->work->auto_prompt_chunk=0;
            return true;
        }
    } catch(const std::exception& e){err=e.what();}
    if(opt.dense && opt.dense->work){opt.dense->work->borrow_failed=true;opt.dense->work->auto_prompt_chunk=0;}
    logits_out.clear();return false;
}

const void* forward_weight(DenseResidency& dense, const std::string& name, int& type) {
    if (!dense.file || dense.slots.empty()) throw std::runtime_error("target weights are not initialized");
    W weight = take(dense.file->gg, dense.slots.back(), &dense.bytes, name);
    type = weight.type;
    return weight.p;
}
IndexerCache* forward_indexer(DenseResidency& dense) {
    return dense.work ? dense.work->idx.get() : nullptr;
}

void forward_commit_device(DenseResidency& dense,const RoundVerdict* verdict,void* stream) {
    if(!dense.work)throw std::runtime_error("target device commit: uninitialized residency");
    if(dense.work->idx)indexer_commit_device(*dense.work->idx,verdict,stream);
}
void forward_commit_mirror(DenseResidency& dense,KvCache& cache,const RoundVerdict& verdict) {
    if(verdict.count<1||verdict.count>8||verdict.next_position>cache.len||verdict.next_position<0)
        throw std::runtime_error("target device commit: invalid verdict");
    cache.len=verdict.next_position;
    if(dense.work && dense.work->idx) {
        auto& c=*dense.work->idx;c.valid_pos=cache.len;c.carry=cache.len%c.kpool;c.n_pool=cache.len/c.kpool;
    }
}


bool DflyCapture::init(const Geometry& g, int rows, std::string& e) {
    release();
    const int n_blocks = (int) g.layers.size() - g.n_mtp_blocks;
    if (rows < 1 || g.idx_key_dim <= 0 || g.idx_kpool <= 0 || g.idx_top_k <= 0) {
        e = "DflyCapture: needs rows >= 1 and a geometry with an indexer";
        return false;
    }
    n_embd = g.hidden_dim; key_dim = g.idx_key_dim; kpool = g.idx_kpool; capacity_rows = rows;
    sel_cap = g.idx_top_k + g.idx_kpool;
    n_idx_layers = 0;
    std::vector<char> sparse((std::size_t) n_blocks, 0);
    for (int b = 0; b < n_blocks; ++b)
        if (g.layers[(std::size_t) b].attention != Attention::Sliding) { sparse[(std::size_t) b] = 1; ++n_idx_layers; }
    for (int b : sel_blocks)
        if (b < 0 || b >= n_blocks || !sparse[(std::size_t) b]) {
            e = "DflyCapture: selection block " + std::to_string(b) + " is not a sparse block";
            release();
            return false;
        }
    auto grab = [&](std::size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        return p;
    };
    hidden = grab((std::size_t) rows * n_embd * 4);
    bool ok = hidden != nullptr;
    for (std::size_t i = 0; ok && i < sel_blocks.size(); ++i) {
        sel.push_back(grab((std::size_t) rows * sel_cap * 4));
        ok = sel.back() != nullptr;
    }
    for (int l = 0; ok && l < n_idx_layers; ++l) {
        vg.push_back(grab((std::size_t) (kpool - 1 + rows) * 2 * key_dim * 4));
        ok = vg.back() != nullptr;
    }
    if (!ok) {
        e = "DflyCapture: out of device memory";
        release();
        return false;
    }
    return true;
}

void DflyCapture::release() {
    if (hidden) cudaFree(hidden);
    for (void* p : sel) if (p) cudaFree(p);
    for (void* p : vg) if (p) cudaFree(p);
    hidden = nullptr; sel.clear(); vg.clear();
    valid = false; capacity_rows = 0; n_idx_layers = 0;
}

bool dfly_target_weights(const DenseResidency& res, DflyTargetWeights& out, std::string& err) {
    out = DflyTargetWeights{};
    if (res.work == nullptr || !res.work->warm || res.slots.empty() || res.slots.back() == nullptr) {
        err = "dfly_target_weights: no whole pass has run through this residency yet";
        return false;
    }
    const auto& m = res.slots.back()->by_name;
    const auto e = m.find("token_embd.weight");
    const auto o = m.find("output.weight");
    if (e == m.end() || e->second.p == nullptr || o == m.end() || o->second.p == nullptr) {
        err = "dfly_target_weights: the residency holds no token_embd/output (run a pass with logits first)";
        return false;
    }
    out.token_embd = e->second.p;
    out.token_embd_type = e->second.type;
    out.token_embd_shape = e->second.shape;
    out.output = o->second.p;
    out.output_type = o->second.type;
    out.output_shape = o->second.shape;
    return true;
}

bool dfly_rewind(DenseResidency& res, KvCache& cache, const DflyCapture& cap, int new_len, std::string& err) {
    if (res.work == nullptr || res.work->idx == nullptr) { err = "dfly_rewind: no indexer cache"; return false; }
    if (!cap.valid || !cap.has_window) { err = "dfly_rewind: the capture holds no verify window"; return false; }
    IndexerCache& c = *res.work->idx;
    const int end = cap.pos_begin + cap.n_rows;
    if (new_len < cap.pos_begin || new_len > end) {
        err = "dfly_rewind: " + std::to_string(new_len) + " is outside [" + std::to_string(cap.pos_begin) + "," +
              std::to_string(end) + "]";
        return false;
    }
    if (cache.len != end || c.valid_pos != end) {
        err = "dfly_rewind: the cache is not at the end of the pass this capture came from";
        return false;
    }
    if (new_len != end && !indexer_restore_window(c, cap.vg, cap.win_start, end, new_len, err)) return false;
    cache.len = new_len;
    return true;
}

DflyResetResult dfly_abort_reset(DenseResidency& res, KvCache& cache, std::string& err) {
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        err = "dfly_abort_reset: the device did not drain; restart the worker";
        return DflyResetResult::restart_worker;
    }
    if (res.work != nullptr) res.work->invalidate_verify_graphs();
    if (res.work != nullptr && res.work->idx != nullptr && !indexer_force_reset(*res.work->idx, err))
        return DflyResetResult::restart_worker;
    cache.len = 0;
    return DflyResetResult::clean;
}

bool dfly_indexer_snapshot(const DenseResidency& res, std::vector<unsigned char>& out, std::string& err) {
    out.clear();
    if (res.work == nullptr || res.work->idx == nullptr) { err = "dfly_indexer_snapshot: no indexer cache"; return false; }
    const IndexerCache& c = *res.work->idx;
    auto put = [&out](const void* p, std::size_t n) {
        const unsigned char* b = (const unsigned char*) p;
        out.insert(out.end(), b, b + n);
    };
    const int32_t host[3] = {c.n_pool, c.carry, c.valid_pos};
    put(host, sizeof host);
    int32_t dev[2] = {-1, -1};
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(dev, c.counters, sizeof dev, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        err = "dfly_indexer_snapshot: the device counters could not be read";
        return false;
    }
    put(dev, sizeof dev);
    std::vector<unsigned char> tmp;
    for (int l = 0; l < c.n_layers; ++l) {
        const std::size_t pb = (std::size_t) c.n_pool * c.key_dim * 4, vb = (std::size_t) c.carry * 2 * c.key_dim * 4;
        tmp.resize(std::max(pb, vb));
        if (pb && cudaMemcpy(tmp.data(), c.pooled[(std::size_t) l], pb, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError(); err = "dfly_indexer_snapshot: pooled read failed"; return false;
        }
        put(tmp.data(), pb);
        if (vb && cudaMemcpy(tmp.data(), c.vg[(std::size_t) l], vb, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError(); err = "dfly_indexer_snapshot: vg read failed"; return false;
        }
        put(tmp.data(), vb);
    }
    return true;
}

#include "device_loop.inc"
}  // namespace strata::hy4_flash
