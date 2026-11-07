// Force-included after hip_compat/cuda_runtime.h on the HIP copies only.
// That header maps cudaSetDevice onto hipSetDevice, which takes a HIP index.
// These copies are called with the hybrid ordinal (CUDA count + HIP index).
#pragma once
#if defined(STRATA_HYBRID) && defined(__HIPCC__)

#undef cudaSetDevice
#undef cudaGetDevice
#undef cudaGetDeviceProperties
#undef cudaDeviceGetAttribute
#undef cudaMalloc
#undef cudaFree

extern "C" int strata_hybrid_hip_index(int ordinal);
extern "C" void strata_hybrid_note_device(int ordinal);
extern "C" int strata_hybrid_current_ordinal(void);
extern "C" int strata_hybrid_hip_malloc(void** p, size_t n);
extern "C" int strata_hybrid_hip_free(void* p);

static inline hipError_t strata_hybrid_hip_set_device(int ordinal) {
    const int hip = strata_hybrid_hip_index(ordinal);
    if (hip < 0) return hipErrorInvalidDevice;
    const hipError_t e = hipSetDevice(hip);
    if (e == hipSuccess) strata_hybrid_note_device(ordinal);
    return e;
}
#define cudaSetDevice(d) strata_hybrid_hip_set_device(d)

static inline hipError_t strata_hybrid_hip_get_device(int* out) {
    if (out == nullptr) return hipErrorInvalidValue;
    *out = strata_hybrid_current_ordinal();
    return hipSuccess;
}
#define cudaGetDevice(p) strata_hybrid_hip_get_device(p)

static inline hipError_t strata_hybrid_hip_get_props(hipDeviceProp_t* p, int ordinal) {
    const int hip = strata_hybrid_hip_index(ordinal);
    if (p == nullptr || hip < 0) return hipErrorInvalidDevice;
    return hipGetDeviceProperties(p, hip);
}
#define cudaGetDeviceProperties(p, d) strata_hybrid_hip_get_props((p), (d))

static inline hipError_t strata_hybrid_hip_get_attr(int* v, hipDeviceAttribute_t attr, int ordinal) {
    const int hip = strata_hybrid_hip_index(ordinal);
    if (v == nullptr || hip < 0) return hipErrorInvalidDevice;
    return hipDeviceGetAttribute(v, attr, hip);
}
#define cudaDeviceGetAttribute(v, a, d) strata_hybrid_hip_get_attr((v), (a), (d))

#define cudaMalloc(p, n) ((hipError_t) strata_hybrid_hip_malloc((void**) (p), (size_t) (n)))
#define cudaFree(p) ((hipError_t) strata_hybrid_hip_free((void*) (p)))

#endif
