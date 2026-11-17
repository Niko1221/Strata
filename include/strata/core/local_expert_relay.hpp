// Experimental Linux relay for prompt expert uploads on a dual RTX 2080 Ti NVLink system.
// It is intended for asymmetric host PCIe links where CUDA0 (receiver) is narrower/slower than CUDA1 (donor);
// symmetric or reversed link layouts are not performance-validated. The opt-in remains off by default.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(STRATA_LOCAL_CUDA_RELAY_BUILD) && defined(__linux__) && !defined(STRATA_USE_HIP) && \
    !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL) && !defined(SYCL_LANGUAGE_VERSION)
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "strata/core/on_device.hpp"

#define STRATA_LOCAL_CUDA_RELAY 1

namespace strata::core {

class LocalExpertRelay {
    static constexpr size_t kSlotBytes = 4u * 1024u * 1024u;
    static constexpr int kSlots = 2;
    static constexpr unsigned int kNvmlLinkLimit = 18;

    struct NvmlPciInfo {
        char bus_id_legacy[16];
        unsigned int domain;
        unsigned int bus;
        unsigned int device;
        unsigned int pci_device_id;
        unsigned int pci_subsystem_id;
        char bus_id[32];
    };
    static_assert(sizeof(NvmlPciInfo) == 68, "nvmlPciInfo_t ABI layout changed");

    using NvmlInitFn = int (*)();
    using NvmlHandleByPciFn = int (*)(const char*, void**);
    using NvmlLinkStateFn = int (*)(void*, unsigned int, int*);
    using NvmlRemotePciFn = int (*)(void*, unsigned int, NvmlPciInfo*);

    unsigned char* slots_[kSlots]{};
    cudaStream_t upload_ = nullptr;
    cudaEvent_t ready_[kSlots]{}, done_[kSlots]{}, before_ = nullptr;
    bool used_[kSlots]{}, active_ = false;
    size_t next_ = 0, copies_ = 0, bytes_ = 0;

    static cudaError_t enable_peer(int other) {
        const cudaError_t rc = cudaDeviceEnablePeerAccess(other, 0);
        if (rc == cudaErrorPeerAccessAlreadyEnabled) {
            (void) cudaGetLastError();
            return cudaSuccess;
        }
        return rc;
    }

    static bool same_bus_id(const char* a, const char* b) {
        unsigned int ad = 0, ab = 0, as = 0, af = 0;
        unsigned int bd = 0, bb = 0, bs = 0, bf = 0;
        return a != nullptr && b != nullptr &&
               std::sscanf(a, "%x:%x:%x.%x", &ad, &ab, &as, &af) == 4 &&
               std::sscanf(b, "%x:%x:%x.%x", &bd, &bb, &bs, &bf) == 4 &&
               ad == bd && ab == bb && as == bs && af == bf;
    }

    static bool has_active_nvlink_to(void* device, const char* peer_bus_id,
                                     NvmlLinkStateFn get_state, NvmlRemotePciFn get_remote) {
        for (unsigned int link = 0; link < kNvmlLinkLimit; ++link) {
            int active = 0;
            if (get_state(device, link, &active) != 0 || active == 0) continue;
            NvmlPciInfo remote{};
            if (get_remote(device, link, &remote) == 0 && same_bus_id(remote.bus_id, peer_bus_id)) return true;
        }
        return false;
    }

    static bool nvlink_pair_connected(const char* bus0, const char* bus1) {
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return false;
        const auto sym = [lib](const char* name) { return dlsym(lib, name); };
        const auto init = reinterpret_cast<NvmlInitFn>(sym("nvmlInit_v2"));
        const auto shutdown = reinterpret_cast<NvmlInitFn>(sym("nvmlShutdown"));
        const auto by_pci = reinterpret_cast<NvmlHandleByPciFn>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
        const auto get_state = reinterpret_cast<NvmlLinkStateFn>(sym("nvmlDeviceGetNvLinkState"));
        const auto get_remote = reinterpret_cast<NvmlRemotePciFn>(sym("nvmlDeviceGetNvLinkRemotePciInfo_v2"));
        bool connected = false;
        bool initialized = false;
        if (init != nullptr && by_pci != nullptr && get_state != nullptr && get_remote != nullptr && init() == 0) {
            initialized = true;
            void* device0 = nullptr;
            void* device1 = nullptr;
            if (by_pci(bus0, &device0) == 0 && device0 != nullptr &&
                by_pci(bus1, &device1) == 0 && device1 != nullptr) {
                connected = has_active_nvlink_to(device0, bus1, get_state, get_remote) &&
                            has_active_nvlink_to(device1, bus0, get_state, get_remote);
            }
        }
        if (initialized && shutdown != nullptr) (void) shutdown();
        dlclose(lib);
        return connected;
    }

    void destroy() {
        bool resources = before_ != nullptr || upload_ != nullptr;
        for (int i = 0; i < kSlots; ++i)
            resources = resources || slots_[i] != nullptr || ready_[i] != nullptr || done_[i] != nullptr;
        if (!resources) return;   // off: do not create a CUDA context just to destroy an unused relay
        if (upload_ != nullptr) {
            const OnDevice on(1);
            (void) cudaStreamSynchronize(upload_);
        }
        {
            const OnDevice on(0);
            for (int i = 0; i < kSlots; ++i) {
                if (used_[i] && done_[i] != nullptr) (void) cudaEventSynchronize(done_[i]);
                if (done_[i] != nullptr) (void) cudaEventDestroy(done_[i]);
                if (before_ != nullptr && i == 0) (void) cudaEventDestroy(before_);
            }
            before_ = nullptr;
        }
        {
            const OnDevice on(1);
            for (int i = 0; i < kSlots; ++i) {
                if (ready_[i] != nullptr) (void) cudaEventDestroy(ready_[i]);
                if (slots_[i] != nullptr) (void) cudaFree(slots_[i]);
                ready_[i] = nullptr;
                slots_[i] = nullptr;
            }
            if (upload_ != nullptr) (void) cudaStreamDestroy(upload_);
            upload_ = nullptr;
        }
        active_ = false;
    }

public:
    LocalExpertRelay() = default;
    LocalExpertRelay(const LocalExpertRelay&) = delete;
    LocalExpertRelay& operator=(const LocalExpertRelay&) = delete;
    ~LocalExpertRelay() { destroy(); }

    bool active() const { return active_; }
    size_t copies() const { return copies_; }
    size_t bytes() const { return bytes_; }

    // Default off. The opt-in is restricted to two visible RTX 2080 Ti cards (SM75) with an active NVLink in both directions.
    cudaError_t open(const char* flag = "STRATA_LOCAL_PREFILL_RELAY") {
        const char* value = std::getenv(flag);
        if (value == nullptr || value[0] != '1') return cudaSuccess;

        int count = 0;
        cudaError_t rc = cudaGetDeviceCount(&count);
        if (rc != cudaSuccess) return rc;
        if (count != 2) {
            std::fprintf(stderr, "strata local: %s requires exactly two CUDA-visible GPUs (found %d)\n", flag, count);
            return cudaErrorNotSupported;
        }
        for (int dev = 0; dev < 2; ++dev) {
            cudaDeviceProp prop{};
            if ((rc = cudaGetDeviceProperties(&prop, dev)) != cudaSuccess) return rc;
            if (std::strstr(prop.name, "GeForce RTX 2080 Ti") == nullptr || prop.major != 7 || prop.minor != 5) {
                std::fprintf(stderr, "strata local: %s supports only GeForce RTX 2080 Ti / SM75 (CUDA%d is %s, SM%d%d)\n",
                             flag, dev, prop.name, prop.major, prop.minor);
                return cudaErrorNotSupported;
            }
        }

        char bus0[32]{}, bus1[32]{};
        if ((rc = cudaDeviceGetPCIBusId(bus0, sizeof bus0, 0)) != cudaSuccess ||
            (rc = cudaDeviceGetPCIBusId(bus1, sizeof bus1, 1)) != cudaSuccess) return rc;
        if (!nvlink_pair_connected(bus0, bus1)) {
            std::fprintf(stderr, "strata local: %s needs an active NVLink pair between CUDA0 [%s] and CUDA1 [%s]; "
                                 "NVML topology check failed\n", flag, bus0, bus1);
            return cudaErrorNotSupported;
        }

        int can01 = 0, can10 = 0;
        if ((rc = cudaDeviceCanAccessPeer(&can01, 0, 1)) != cudaSuccess ||
            (rc = cudaDeviceCanAccessPeer(&can10, 1, 0)) != cudaSuccess) return rc;
        if (!can01 || !can10) return cudaErrorPeerAccessUnsupported;
        {
            const OnDevice on(0);
            if ((rc = enable_peer(1)) != cudaSuccess ||
                (rc = cudaEventCreateWithFlags(&before_, cudaEventDisableTiming)) != cudaSuccess) return rc;
            for (cudaEvent_t& event : done_)
                if ((rc = cudaEventCreateWithFlags(&event, cudaEventDisableTiming)) != cudaSuccess) return rc;
        }
        {
            const OnDevice on(1);
            if ((rc = enable_peer(0)) != cudaSuccess ||
                (rc = cudaStreamCreateWithFlags(&upload_, cudaStreamNonBlocking)) != cudaSuccess) return rc;
            for (int i = 0; i < kSlots; ++i) {
                if ((rc = cudaMalloc(&slots_[i], kSlotBytes)) != cudaSuccess ||
                    (rc = cudaEventCreateWithFlags(&ready_[i], cudaEventDisableTiming)) != cudaSuccess) return rc;
            }
        }
        active_ = true;
        std::fprintf(stderr, "strata local: %s active: RAM -> CUDA1 [%s] -> CUDA0 [%s]; %d x 4 MiB (8 MiB total)\n",
                     flag, bus1, bus0, kSlots);
        return cudaSuccess;
    }

    cudaError_t begin(cudaStream_t receiver, int device) {
        if (!active_ || device != 0) return cudaSuccess;
        cudaError_t rc = cudaEventRecord(before_, receiver);
        if (rc != cudaSuccess) return rc;
        const OnDevice on(1);
        return cudaStreamWaitEvent(upload_, before_, 0);
    }

    // Host source storage must remain alive until the receiver stream passes this copy's completion point.
    cudaError_t copy(void* dst, const void* src, size_t bytes, cudaStream_t receiver, int device) {
        if (!active_ || device != 0)
            return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, receiver);
        if (dst == nullptr || src == nullptr || receiver == nullptr || bytes == 0) return cudaErrorInvalidValue;

        const OnDevice donor(1);
        for (size_t offset = 0; offset < bytes; offset += kSlotBytes) {
            const int slot = (int) (next_++ % (size_t) kSlots);
            const size_t amount = std::min(kSlotBytes, bytes - offset);
            cudaError_t rc = cudaSuccess;
            if (used_[slot]) rc = cudaStreamWaitEvent(upload_, done_[slot], 0);
            if (rc == cudaSuccess)
                rc = cudaMemcpyAsync(slots_[slot], (const unsigned char*) src + offset, amount,
                                     cudaMemcpyHostToDevice, upload_);
            if (rc == cudaSuccess) rc = cudaEventRecord(ready_[slot], upload_);
            if (rc == cudaSuccess) {
                const OnDevice receiver_device(0);
                rc = cudaStreamWaitEvent(receiver, ready_[slot], 0);
                if (rc == cudaSuccess)
                    rc = cudaMemcpyPeerAsync((unsigned char*) dst + offset, 0, slots_[slot], 1, amount, receiver);
                if (rc == cudaSuccess) rc = cudaEventRecord(done_[slot], receiver);
            }
            if (rc != cudaSuccess) {
                // Keep the caller's host source alive until any accepted H2D has finished before returning the error.
                (void) cudaStreamSynchronize(upload_);
                (void) cudaStreamSynchronize(receiver);
                return rc;
            }
            used_[slot] = true;
        }
        ++copies_;
        bytes_ += bytes;
        return cudaSuccess;
    }
};

// Owned by generate(), opened before cache sizing; used only on CUDA0's primary prefill upload stream.
inline LocalExpertRelay* local_prefill_relay = nullptr;

}  // namespace strata::core
#endif
