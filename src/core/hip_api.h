#pragma once
// HIP runtime entry points. The CUDA wrappers call these so the two
// platform headers never land in one translation unit.
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct HhDevice {
    char name[256];
    char arch[64];
    size_t total_mem;
    size_t shared_mem;
    int warp;
    int sms;
    int max_threads;
    int clock_khz;
    int l2;
    int integrated;
    int pci_bus;
    int pci_dev;
};

int hh_device_count(void);
int hh_devices(struct HhDevice* out, int cap);
int hh_get_device(int* index);
int hh_set_device(int index);
int hh_set_device_flags(unsigned flags);
int hh_sync(void);
int hh_malloc(void** p, size_t n);
int hh_free(void* p);
int hh_host_malloc(void** p, size_t n, unsigned int flags);
int hh_host_free(void* p);
int hh_host_register(void* p, size_t n, unsigned int flags);
int hh_host_unregister(void* p);
int hh_host_device_pointer(void** dev, void* host, unsigned int flags);
int hh_memcpy(void* dst, const void* src, size_t n, int kind);
int hh_memcpy_async(void* dst, const void* src, size_t n, int kind, void* stream);
int hh_memcpy_2d_async(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h, int kind, void* stream);
int hh_memset(void* p, int v, size_t n);
int hh_memset_async(void* p, int v, size_t n, void* stream);
int hh_mem_info(size_t* free_b, size_t* total_b);
int hh_launch(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, void** args,
              size_t shared, void* stream);
int hh_launch_host(void* stream, void (*fn)(void*), void* data);
int hh_stream_create(void** s, unsigned int flags, int priority, int with_priority);
int hh_stream_destroy(void* s);
int hh_stream_sync(void* s);
int hh_stream_query(void* s);
int hh_stream_begin_capture(void* s, int mode);
int hh_stream_end_capture(void* s, void** graph);
int hh_stream_is_capturing(void* s, int* status);
int hh_stream_capture_info(void* s, int* status, unsigned long long* id, void** graph, const void*** deps, size_t* ndeps);
int hh_stream_wait_event(void* s, void* event, unsigned int flags);
int hh_event_create(void** ev, unsigned int flags, int with_flags);
int hh_event_destroy(void* ev);
int hh_event_record(void* ev, void* s);
int hh_event_sync(void* ev);
int hh_event_query(void* ev);
int hh_event_elapsed(float* ms, void* a, void* b);
int hh_graph_destroy(void* g);
int hh_graph_exec_destroy(void* g);
int hh_graph_instantiate(void** exec, void* graph, unsigned long long flags);
int hh_graph_launch(void* exec, void* s);
int hh_graph_upload(void* exec, void* s);
int hh_graph_nodes(void* g, void** nodes, size_t* n);
int hh_graph_edges(void* g, void** from, void** to, size_t* n);
int hh_graph_node_type(void* node, int* type);
int hh_graph_kernel_params(void* node, void** fn, unsigned* gx, unsigned* gy, unsigned* gz, unsigned* bx, unsigned* by,
                           unsigned* bz, unsigned* shared, void*** kparams, void*** extra);
int hh_func_attrs(void* fn, size_t* shared, size_t* cnst, size_t* local, int* max_threads, int* regs, int* ptx, int* binary,
                  int* cache_ca, int* max_dynamic, int* carveout);
int hh_func_set_attr(void* fn, int attr, int value);
int hh_symbol_address(void** dev, const void* symbol);
int hh_device_attr(int* value, int attr, int index);
int hh_stream_priority_range(int* least, int* greatest);
int hh_driver_version(int* v);
int hh_runtime_version(int* v);
int hh_last_error(void);
int hh_peek_error(void);
const char* hh_error_string(int e);
int hh_occupancy(int* blocks, void* fn, int block, size_t smem, unsigned int flags);

#ifndef STRATA_HYBRID_NO_BLAS
int hh_blas_create(void** h);
int hh_blas_destroy(void* h);
int hh_blas_set_stream(void* h, void* s);
int hh_blas_set_math(void* h, int mode);
int hh_blas_set_workspace(void* h, void* workspace, size_t bytes);
int hh_blas_sgemm(void* h, int ta, int tb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B,
                  int ldb, const float* beta, float* C, int ldc);
int hh_blas_gemm_ex(void* h, int ta, int tb, int m, int n, int k, const void* alpha, const void* A, int at, int lda,
                    const void* B, int bt, int ldb, const void* beta, void* C, int ct, int ldc, int compute, int algo);
int hh_blas_gemm_strided(void* h, int ta, int tb, int m, int n, int k, const void* alpha, const void* A, int at, int lda,
                         long long stride_a, const void* B, int bt, int ldb, long long stride_b, const void* beta, void* C,
                         int ct, int ldc, long long stride_c, int batch, int compute, int algo);
#endif

#ifdef __cplusplus
}
#endif
