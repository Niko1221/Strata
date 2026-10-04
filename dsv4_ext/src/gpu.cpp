// dsv4/gpu.cpp - CPU-emulated device layer (see include/dsv4/gpu.hpp).
// "VRAM" is host memory; H2D/D2H are memcpy; the kernels are the same ggml-identical row math used by
// the CPU path, so HIT and MISS produce bit-identical results and the residency logic is testable
// without a GPU. The real device lives in src/gpu_cuda.cu and defines the very same symbols: a binary
// compiles exactly ONE of the two files, never both.
#include "dsv4/gpu.hpp"

#include "dsv4/dequant.hpp"
#include "dsv4/ops.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dsv4 {
namespace gpu {
namespace {

size_t g_total = 0, g_used = 0;
bool g_ready = false;

}  // namespace

bool is_emulated() { return true; }

bool init(std::string& err) {
    if (g_ready) return true;
    const char* env = std::getenv("DSV4_EMULATED_VRAM_MIB");
    const long long mib = env ? std::atoll(env) : 8192;
    if (mib <= 0) { err = "DSV4_EMULATED_VRAM_MIB must be positive"; return false; }
    g_total = (size_t) mib << 20;
    g_used = 0;
    g_ready = true;
    return true;
}

void* alloc(size_t n) {
    if (!g_ready || g_used + n > g_total) return nullptr;
    void* p = std::malloc(n);
    if (!p) return nullptr;
    g_used += n;
    return p;
}

void release(void* p) { std::free(p); }  // g_used is never decreased: the pool is freed with the model

void mem_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes) *free_bytes = g_total > g_used ? g_total - g_used : 0;
    if (total_bytes) *total_bytes = g_total;
}

void h2d(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }
void d2h(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

// The emulated device decodes with src/dequant.cpp, so it can handle every type that file supports.
bool type_supported(uint32_t type) { return g_ready && dequant_supported(type); }
bool experts_supported(uint32_t g, uint32_t u, uint32_t d) {
    return type_supported(g) && type_supported(u) && type_supported(d);
}

// "Device" memory IS host memory here, so a host expert needs no staging at all: accept the request
// and let experts_hit() read the host pointers directly. Registering a pool would only burn RAM.
bool staging(uint8_t*, size_t) { return g_ready; }
bool has_staging() { return g_ready; }

bool matvec(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* d_x, float* d_y) {
    const size_t rb = row_bytes(type, in);
    if (!rb) return false;
    if (!type_supported(type)) return false;   // the caller falls back to the host path
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) d_y[r] = row_dot(type, W + (size_t) r * rb, d_x, in);
    return true;
}

bool experts_hit(const ExpPtrs& p, uint32_t type_g, uint32_t type_u, uint32_t type_d, int64_t ff, int64_t dim,
                 float swiglu_limit, const float* d_x, float* d_g, float* d_u, float* d_a, float* d_y) {
    if (p.n <= 0) return true;
    const size_t rg = row_bytes(type_g, dim), ru = row_bytes(type_u, dim), rd = row_bytes(type_d, ff);
    if (!rg || !ru || !rd) return false;
    if (!experts_supported(type_g, type_u, type_d)) return false;
    std::memset(d_y, 0, (size_t) dim * 4);
    std::vector<float> y((size_t) dim);
    for (int k = 0; k < p.n && k < kMaxHit; ++k) {
        // A host expert is already where this backend would have to copy it to.
        const uint8_t* gsrc = p.gate[k] ? p.gate[k] : p.hg[k];
        const uint8_t* usrc = p.up[k] ? p.up[k] : p.hu[k];
        const uint8_t* dsrc = p.down[k] ? p.down[k] : p.hd[k];
        if (!gsrc || !usrc || !dsrc) return false;
        float* g = d_g + (size_t) k * ff;
        float* u = d_u + (size_t) k * ff;
        float* a = d_a + (size_t) k * ff;
        matvec(type_g, gsrc, ff, dim, d_x, g);
        matvec(type_u, usrc, ff, dim, d_x, u);
        swiglu_clamped(g, u, (int) ff, swiglu_limit, a);
        for (int64_t i = 0; i < ff; ++i) a[i] *= p.w[k];
        matvec(type_d, dsrc, dim, ff, a, y.data());
        for (int64_t i = 0; i < dim; ++i) d_y[i] += y[(size_t) i];
    }
    return true;
}

}  // namespace gpu
}  // namespace dsv4
