// src/core/expert_cache.cpp - R4's slot storage, residency table, and R4.3 elastic VMM tier.
#include "strata/core/expert_cache.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

namespace strata::core {

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    (void) version;
    return true;
}

ExpertCache::~ExpertCache() { close(); }

bool ExpertCache::vmm_supported() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    if (cuInit(0) != CUDA_SUCCESS) { cached = 0; return false; }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, dev) != CUDA_SUCCESS) { cached = 0; return false; }
    int vmm = 0;
    if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cuDev) != CUDA_SUCCESS) {
        cached = 0;
        return false;
    }
    cached = (vmm != 0) ? 1 : 0;
    return cached == 1;
}

size_t ExpertCache::vmm_granularity() {
    if (!vmm_supported()) return 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, dev) != CUDA_SUCCESS) return 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
        if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
            return 0;
        }
    }
    return gran;
}

bool ExpertCache::open_vmm(int64_t total_bytes, int64_t n_layers, int64_t n_expert, std::string& err) {
    (void) n_layers;
    (void) n_expert;
    close();
    size_t gran = vmm_granularity();
    if (gran == 0) {
        err = "ExpertCache: VMM not supported or granularity query failed";
        return false;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    CUdevice cuDev = 0;
    if (cuDeviceGet(&cuDev, dev) != CUDA_SUCCESS) {
        err = "ExpertCache: cuDeviceGet failed";
        return false;
    }

    const size_t aligned_size = ((size_t) total_bytes + gran - 1) / gran * gran;

    CUdeviceptr va = 0;
    CUresult res = cuMemAddressReserve(&va, aligned_size, 0, 0, 0);
    if (res != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(res, &s);
        err = std::string("ExpertCache: cuMemAddressReserve failed: ") + (s ? s : "unknown");
        return false;
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = dev;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    const size_t chunk_size = gran;
    const size_t num_chunks = aligned_size / chunk_size;

    vmm_chunks_.reserve(num_chunks);
    size_t mapped = 0;

    for (size_t i = 0; i < num_chunks; ++i) {
        CUmemGenericAllocationHandle handle = 0;
        res = cuMemCreate(&handle, chunk_size, &prop, 0);
        if (res != CUDA_SUCCESS) {
            for (auto& c : vmm_chunks_) {
                cuMemUnmap((CUdeviceptr) (va + c.offset), (size_t) c.size);
                cuMemRelease((CUmemGenericAllocationHandle) c.handle);
            }
            vmm_chunks_.clear();
            cuMemAddressFree(va, aligned_size);
            const char* s = nullptr;
            cuGetErrorString(res, &s);
            err = std::string("ExpertCache: cuMemCreate failed: ") + (s ? s : "out of VRAM");
            return false;
        }

        res = cuMemMap(va + (size_t) mapped, chunk_size, 0, handle, 0);
        if (res != CUDA_SUCCESS) {
            cuMemRelease(handle);
            for (auto& c : vmm_chunks_) {
                cuMemUnmap((CUdeviceptr) (va + c.offset), (size_t) c.size);
                cuMemRelease((CUmemGenericAllocationHandle) c.handle);
            }
            vmm_chunks_.clear();
            cuMemAddressFree(va, aligned_size);
            const char* s = nullptr;
            cuGetErrorString(res, &s);
            err = std::string("ExpertCache: cuMemMap failed: ") + (s ? s : "unknown");
            return false;
        }

        res = cuMemSetAccess(va + (size_t) mapped, chunk_size, &desc, 1);
        if (res != CUDA_SUCCESS) {
            cuMemUnmap(va + (size_t) mapped, chunk_size);
            cuMemRelease(handle);
            for (auto& c : vmm_chunks_) {
                cuMemUnmap((CUdeviceptr) (va + c.offset), (size_t) c.size);
                cuMemRelease((CUmemGenericAllocationHandle) c.handle);
            }
            vmm_chunks_.clear();
            cuMemAddressFree(va, aligned_size);
            const char* s = nullptr;
            cuGetErrorString(res, &s);
            err = std::string("ExpertCache: cuMemSetAccess failed: ") + (s ? s : "unknown");
            return false;
        }

        vmm_chunks_.push_back({(uint64_t) handle, (uint64_t) mapped, (uint64_t) chunk_size});
        mapped += chunk_size;
    }

    base_ = (uint8_t*) va;
    vmm_va_ = (uint64_t) va;
    vmm_va_size_ = (uint64_t) aligned_size;
    max_bytes_ = (int64_t) aligned_size;
    mapped_bytes_ = (int64_t) mapped;
    vmm_ = true;

    if (cudaMemset(base_, 0, (size_t) mapped_bytes_) != cudaSuccess) {
        err = "ExpertCache: cudaMemset of VMM slot arena failed";
        close();
        return false;
    }

    return true;
}

void ExpertCache::close_vmm() {
    if (vmm_va_ != 0) {
        for (auto& c : vmm_chunks_) {
            cuMemUnmap((CUdeviceptr) (vmm_va_ + c.offset), (size_t) c.size);
            cuMemRelease((CUmemGenericAllocationHandle) c.handle);
        }
        vmm_chunks_.clear();
        cuMemAddressFree((CUdeviceptr) vmm_va_, (size_t) vmm_va_size_);
        vmm_va_ = 0;
        vmm_va_size_ = 0;
    }
    base_ = nullptr;
    vmm_ = false;
    max_slots_ = 0;
    max_bytes_ = 0;
    mapped_bytes_ = 0;
}

int64_t ExpertCache::shrink(int64_t keep_bytes, std::string& err) {
    (void) err;
    if (!vmm_ || vmm_chunks_.empty()) return 0;
    cudaDeviceSynchronize();

    size_t gran = vmm_granularity();
    if (gran == 0) return 0;
    const size_t keep_aligned = ((size_t) std::max<int64_t>(0, keep_bytes) + gran - 1) / gran * gran;
    if (keep_aligned >= (size_t) mapped_bytes_) return 0;

    int64_t freed = 0;
    while (!vmm_chunks_.empty() && vmm_chunks_.back().offset >= keep_aligned) {
        auto& c = vmm_chunks_.back();
        cuMemUnmap((CUdeviceptr) (vmm_va_ + c.offset), (size_t) c.size);
        cuMemRelease((CUmemGenericAllocationHandle) c.handle);
        freed += (int64_t) c.size;
        mapped_bytes_ -= (int64_t) c.size;
        vmm_chunks_.pop_back();
    }

    int64_t new_slots = 0;
    if (!off_.empty()) {
        while (new_slots < (int64_t) off_.size() - 1 && (int64_t) off_[(size_t) (new_slots + 1)] <= mapped_bytes_) {
            ++new_slots;
        }
    } else if (blob_ > 0) {
        new_slots = mapped_bytes_ / blob_;
    }

    for (size_t i = 0; i < residency_.size(); ++i) {
        if (residency_[i] >= new_slots && residency_[i] < slots_) {
            residency_[i] = kNotResident;
        }
    }
    slots_ = new_slots;
    return freed;
}

int64_t ExpertCache::grow(int64_t new_bytes, const std::vector<std::pair<int32_t, int32_t>>& profile,
                          BlobFn blob_fn, void* blob_user, BlobSizeFn size_fn, void* size_user, std::string& err) {
    if (!vmm_ || vmm_va_ == 0) return 0;
    cudaDeviceSynchronize();

    size_t gran = vmm_granularity();
    if (gran == 0) return 0;
    const size_t target_aligned = std::min((size_t) vmm_va_size_, ((size_t) new_bytes + gran - 1) / gran * gran);
    if (target_aligned <= (size_t) mapped_bytes_) return 0;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = dev;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    const size_t chunk_size = gran;
    while ((size_t) mapped_bytes_ < target_aligned) {
        CUmemGenericAllocationHandle handle = 0;
        CUresult res = cuMemCreate(&handle, chunk_size, &prop, 0);
        if (res != CUDA_SUCCESS) break;

        res = cuMemMap(vmm_va_ + (size_t) mapped_bytes_, chunk_size, 0, handle, 0);
        if (res != CUDA_SUCCESS) {
            cuMemRelease(handle);
            break;
        }

        res = cuMemSetAccess(vmm_va_ + (size_t) mapped_bytes_, chunk_size, &desc, 1);
        if (res != CUDA_SUCCESS) {
            cuMemUnmap(vmm_va_ + (size_t) mapped_bytes_, chunk_size);
            cuMemRelease(handle);
            break;
        }

        vmm_chunks_.push_back({(uint64_t) handle, (uint64_t) mapped_bytes_, (uint64_t) chunk_size});
        mapped_bytes_ += (int64_t) chunk_size;
    }

    int64_t old_slots = slots_;
    int64_t new_slots = old_slots;
    if (!off_.empty()) {
        while (new_slots < (int64_t) off_.size() - 1 && (int64_t) off_[(size_t) (new_slots + 1)] <= mapped_bytes_) {
            ++new_slots;
        }
    } else if (blob_ > 0) {
        new_slots = mapped_bytes_ / blob_;
    }
    slots_ = new_slots;

    int64_t filled = 0;
    if (blob_fn != nullptr) {
        for (int64_t s = old_slots; s < new_slots; ++s) {
            if ((size_t) s < profile.size()) {
                const auto& pr = profile[(size_t) s];
                const int32_t l = pr.first;
                const int32_t e = pr.second;
                const uint8_t* b = blob_fn(l, e, blob_user);
                const int64_t bsz = size_fn ? size_fn(l, size_user) : 0;
                if (b != nullptr && fill_slot_blocking((int32_t) s, b, err, bsz)) {
                    assign_slot((int32_t) s, l, e);
                    ++filled;
                }
            }
        }
    }
    return filled;
}

void ExpertCache::evict_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return;
    for (size_t i = 0; i < residency_.size(); ++i) {
        if (residency_[i] == slot) {
            residency_[i] = kNotResident;
            break;
        }
    }
}

void ExpertCache::assign_slot(int32_t slot, int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    if (slot < 0 || slot >= slots_) return;
    evict_slot(slot);
    residency_[(size_t) (layer * n_expert_ + expert)] = slot;
}

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    bool ok = false;
    if (vmm_supported()) {
        std::string vmm_err;
        ok = open_vmm((int64_t) want, n_layers, n_expert, vmm_err);
        if (!ok) {
            // Log fallback message if VMM failed, then fallback to standard cudaMalloc
            std::fprintf(stderr, "strata: CUDA VMM allocation failed (%s); falling back to cudaMalloc\n",
                         vmm_err.c_str());
        }
    }

    if (!ok) {
        if (cudaMalloc((void**) &base_, (size_t) want) != cudaSuccess) {
            base_ = nullptr;
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
                          (double) want / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
            err = buf;
            return false;
        }
        if (cudaMemset(base_, 0, (size_t) want) != cudaSuccess) {
            err = "ExpertCache: cudaMemset of the slot arena failed";
            close();
            return false;
        }
        vmm_ = false;
        max_bytes_ = (int64_t) want;
        mapped_bytes_ = (int64_t) want;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    max_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;

    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        off[i + 1] = off[i] + ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    max_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    return true;
}

void ExpertCache::close() {
    off_.clear();
    if (vmm_) {
        close_vmm();
    } else if (base_ != nullptr) {
        cudaFree(base_);
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    max_slots_ = 0;
    max_bytes_ = 0;
    mapped_bytes_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpyAsync(dst, host_blob, n, cudaMemcpyHostToDevice,
                                          (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
    const cudaError_t e = cudaMemcpy(dst, host_blob, n, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + cudaGetErrorString(e);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    std::vector<uint8_t> got((size_t) nb);
    const cudaError_t e = cudaMemcpy(got.data(), src, (size_t) nb, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        err = std::string("ExpertCache::verify_slot: ") + cudaGetErrorString(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
