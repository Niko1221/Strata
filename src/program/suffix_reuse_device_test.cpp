// src/program/suffix_reuse_device_test.cpp - T7: the device-path regression for the
// 2026-10-12 SIGSEGV fix.  This test pins the LIVE failing inputs from the bug report
// (`/opt/data/cache/sfx_live_tokens.hpp`):
//
//   edit_pos=3239, old_suffix_len=1, uniform=true, delta=+54
//
// plus the non-uniform deltas case and the null-deltas guard, and proves the splice
// never dereferences a device pointer on the host.
//
// HOW IT WORKS.  The test compiles the splice path with the transport forced on
// (STRATA_SPLICE_TRANSPORT=1, STRATA_SPLICE_FAKE_CUDA=1 in CMake).  The "device" is a
// PROT_NONE mmap region - any direct host access faults, exactly as a discrete-GPU
// address would.  A tiny fake-cuda API (the only legal path across the device) provides
// cudaPointerGetAttributes + cudaMemcpy.  The RED->GREEN property: without the
// transport (e.g. a future refactor that drops the activation), the splice dereferences
// the device pointer directly and SEGVs; with the transport it goes through staging
// memcpys and completes correctly.
//
// CPU-ONLY.  The fake-cuda API is supplied below; we do not include <cuda_runtime.h>.
// The CI loop compiles the test as a regular g++ translation unit and links the same
// Strata headers the production build uses, so the test exercises the EXACT code path
// that ships in the CUDA binary.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

// ---- a tiny fake CUDA runtime (the ONLY legal path across "device" memory) ----
// Defined BEFORE the splice headers so the transport's inlined code calls these stubs.
namespace fake_cuda {
void* dev_base = nullptr;
size_t dev_size = 0;
int memcpy_calls = 0;
int memcpy_d2h_calls = 0;
int memcpy_h2d_calls = 0;
}  // namespace fake_cuda

#define cudaMemoryTypeHost 1
#define cudaMemoryTypeDevice 2
#define cudaMemoryTypeManaged 3
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

struct cudaPointerAttributes { int type; };
using cudaError_t = int;

static inline cudaError_t cudaGetLastError() { return 0; }
static inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr, const void* p) {
    if (fake_cuda::dev_base == nullptr) return 1;
    const char* b = (const char*) fake_cuda::dev_base;
    const char* q = (const char*) p;
    if (q < b || q >= b + fake_cuda::dev_size) return 1;
    attr->type = cudaMemoryTypeDevice;
    return cudaSuccess;
}
static inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, int kind) {
    ++fake_cuda::memcpy_calls;
    if (kind == cudaMemcpyDeviceToHost) ++fake_cuda::memcpy_d2h_calls;
    else if (kind == cudaMemcpyHostToDevice) ++fake_cuda::memcpy_h2d_calls;
    mprotect(fake_cuda::dev_base, fake_cuda::dev_size, PROT_READ | PROT_WRITE);
    std::memcpy(dst, src, n);
    mprotect(fake_cuda::dev_base, fake_cuda::dev_size, PROT_NONE);
    return cudaSuccess;
}

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/program/kv_splice.hpp"
#include "strata/program/kv_splice_apply.hpp"
#include "strata/program/suffix_reuse.hpp"
#include "strata/program/kv_splice_transport.hpp"

using strata::program::suffix_reuse::SuffixHookOutcome;
using strata::program::suffix_reuse::suffix_reuse_apply_invalidation;
using strata::program::suffix_reuse::rotate_qsa_tail_inplace;
using strata::program::suffix_reuse::build_rope_table_host;
using strata::program::suffix_reuse::rotate_k_by_delta;
using strata::program::suffix_reuse::detail::fp32_to_fp16_bits;
using strata::program::suffix_reuse::detail::fp16_bits_to_fp32;
using strata::program::suffix_reuse::transport::is_active;
using strata::program::suffix_reuse::transport::pointer_is_device;
using strata::program::suffix_reuse::SpliceStatus;

namespace {

// ---- result tally ------------------------------------------------------------
int g_fail = 0;
int g_pass = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}

// ---- the device-backed QsaState fixture --------------------------------------
struct DeviceFixture {
    int head_dim = 256, n_rot = 64, n_head_kv = 2;
    int64_t page_size = 4, max_cells = 131072;
    int half = n_rot / 2;
    int64_t pages = max_cells / page_size;
    int64_t pool_rows = pages * n_head_kv * page_size;
    size_t k_bytes = (size_t) pool_rows * head_dim * 2;
    size_t tab_floats = (size_t) max_cells * half;
    size_t total = k_bytes + 2 * tab_floats * 4;
    void* dev = nullptr;
    std::vector<uint16_t> k_shadow;
    std::vector<float> cos_shadow, sin_shadow;
    int64_t edit_pos = 3239;     // LIVE from /opt/data/cache/sfx_live_tokens.hpp
    int64_t old_suffix_len = 1;  // LIVE
    int delta = 54;              // LIVE: 3294 - 3240
    DeviceFixture() {
        dev = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (dev == MAP_FAILED) { std::perror("mmap"); std::exit(99); }
        k_shadow.resize(k_bytes / 2);
        for (size_t i = 0; i < k_shadow.size(); ++i)
            k_shadow[i] = fp32_to_fp16_bits(1.0f + (float) (i % 7));
        cos_shadow.assign(tab_floats, 0.0f);
        sin_shadow.assign(tab_floats, 0.0f);
        build_rope_table_host(n_rot, 10000000.0, (int) max_cells,
                              cos_shadow.data(), sin_shadow.data());
        std::memcpy(dev, k_shadow.data(), k_bytes);
        std::memcpy((char*) dev + k_bytes, cos_shadow.data(), tab_floats * 4);
        std::memcpy((char*) dev + k_bytes + tab_floats * 4, sin_shadow.data(), tab_floats * 4);
        mprotect(dev, total, PROT_NONE);
        fake_cuda::dev_base = dev;
        fake_cuda::dev_size = total;
    }
    ~DeviceFixture() {
        mprotect(dev, total, PROT_READ | PROT_WRITE);
        munmap(dev, total);
    }
    strata::core::QsaState make_state() {
        strata::core::QsaState st{};
        st.k_pool = (uint16_t*) dev;
        st.cos_tab = (float*) ((char*) dev + k_bytes);
        st.sin_tab = (float*) ((char*) dev + k_bytes + tab_floats * 4);
        st.kv_mode = 0;
        st.max_cells = max_cells;
        st.n_slots = pages;
        return st;
    }
    // The "expected" value of cell 0 for row (row_idx, h) after rotation by `delta`.
    uint16_t expected_cell0(int64_t row_idx_in_pool, int delta_arg) {
        const int64_t global_i_0 = row_idx_in_pool * head_dim + 0;
        const int64_t global_i_half = row_idx_in_pool * head_dim + n_rot / 2;
        const float a0 = 1.0f + (float) (global_i_0 % 7);
        const float b0 = 1.0f + (float) (global_i_half % 7);
        const int p_delta = delta_arg < 0 ? -delta_arg : delta_arg;
        const float s_sign = delta_arg < 0 ? -1.0f : 1.0f;
        const float c = cos_shadow[(size_t) p_delta * half + 0];
        const float s = s_sign * sin_shadow[(size_t) p_delta * half + 0];
        const float oa = a0 * c - b0 * s;
        return fp32_to_fp16_bits(oa);
    }
    int64_t row_idx(int h) {
        return (edit_pos / page_size) * n_head_kv * page_size
             + (int64_t) h * page_size + edit_pos % page_size;
    }
    // Reset the device to the original k_shadow contents (so successive tests start fresh).
    void reset_k() {
        mprotect(dev, total, PROT_READ | PROT_WRITE);
        std::memcpy(dev, k_shadow.data(), k_bytes);
        mprotect(dev, total, PROT_NONE);
    }
    // Read a single row back from the device (after the splice).
    void read_row(int h, std::vector<uint16_t>& out) {
        mprotect(dev, total, PROT_READ);
        const int64_t row_idx_in_pool = (edit_pos / page_size) * n_head_kv * page_size
                                      + (int64_t) h * page_size + edit_pos % page_size;
        out.assign(head_dim, 0);
        std::memcpy(out.data(), (const uint16_t*) dev + (size_t) row_idx_in_pool * head_dim,
                    head_dim * sizeof(uint16_t));
        mprotect(dev, total, PROT_NONE);
    }
};

}  // namespace

int main() {
    std::printf("suffix_reuse_device_test (T7: device-path regression for the 2026-10-12 SIGSEGV)\n");

    // ---- T7.0 transport is active in this build ----
    check(is_active(),
          "T7.0: kv_splice_transport is_active() returns true (transport is wired)");

    // ---- T7.1 pointer_is_device on a PROT_NONE region reports device ----
    DeviceFixture f;
    check(pointer_is_device(f.dev),
          "T7.1: pointer_is_device on PROT_NONE device region reports device-backed");

    // ---- T7.2 LIVE case: edit_pos=3239, old_suffix_len=1, uniform delta=54 ----
    {
        fake_cuda::memcpy_calls = 0;
        fake_cuda::memcpy_d2h_calls = 0;
        fake_cuda::memcpy_h2d_calls = 0;
        auto st = f.make_state();
        const bool ok = rotate_qsa_tail_inplace(
            st, f.head_dim, f.n_rot, f.n_head_kv, (int) f.max_cells, f.page_size,
            f.edit_pos, f.old_suffix_len, /*uniform=*/true, f.delta,
            /*deltas=*/nullptr, /*skip_rotation=*/false, f.pool_rows);
        check(ok, "T7.2 LIVE: rotate_qsa_tail_inplace returns true on device-backed state");
        check(fake_cuda::memcpy_d2h_calls >= 3,
              "T7.2 LIVE: staging copied at least 3 ranges D2H (cos+sin slice, k range)");
        check(fake_cuda::memcpy_h2d_calls >= 1,
              "T7.2 LIVE: staging committed the k range back H2D");
        // The rotated band [0, n_rot) of BOTH heads must match the expected rotation.
        int bad = 0;
        for (int h = 0; h < f.n_head_kv; ++h) {
            std::vector<uint16_t> row;
            f.read_row(h, row);
            const int64_t r = f.row_idx(h);
            const uint16_t want = f.expected_cell0(r, f.delta);
            if (row[0] != want) ++bad;
        }
        check(bad == 0,
              "T7.2 LIVE: cell 0 of both heads matches the expected rotation (uniform delta=+54)");
        // The untouched band [n_rot, head_dim) must be byte-identical to the original.
        int untouched_bad = 0;
        for (int h = 0; h < f.n_head_kv; ++h) {
            std::vector<uint16_t> row;
            f.read_row(h, row);
            const int64_t r = f.row_idx(h);
            for (int i = f.n_rot; i < f.head_dim; ++i) {
                const int64_t global_i = r * f.head_dim + i;
                const uint16_t orig = fp32_to_fp16_bits(1.0f + (float) (global_i % 7));
                if (row[i] != orig) ++untouched_bad;
            }
        }
        check(untouched_bad == 0,
              "T7.2 LIVE: untouched band [n_rot, head_dim) is byte-identical to original "
              "(splice only rotated [0, n_rot))");
    }

    // ---- T7.3 non-uniform deltas: rotate the band by per-row offsets ----
    {
        f.reset_k();
        fake_cuda::memcpy_calls = 0;
        auto st = f.make_state();
        const int64_t edit_pos_nu = 4100;
        const int64_t old_suffix_len_nu = 4;
        const int64_t deltas_nu[4] = { 10, -3, 7, 0 };
        const bool ok = rotate_qsa_tail_inplace(
            st, f.head_dim, f.n_rot, f.n_head_kv, (int) f.max_cells, f.page_size,
            edit_pos_nu, old_suffix_len_nu, /*uniform=*/false, /*delta=*/0,
            deltas_nu,    // already int64_t
            /*skip_rotation=*/false, f.pool_rows);
        check(ok, "T7.3 non-uniform: rotate_qsa_tail_inplace returns true with per-row deltas");
        check(fake_cuda::memcpy_calls > 0,
              "T7.3 non-uniform: staging memcpys fired (transport still active for non-uniform)");
        int bad = 0;
        mprotect(f.dev, f.total, PROT_READ);
        for (int k = 0; k < (int) old_suffix_len_nu; ++k) {
            const int64_t p = edit_pos_nu + k;
            const int64_t d = deltas_nu[k];
            for (int h = 0; h < f.n_head_kv; ++h) {
                const int64_t r = (p / f.page_size) * f.n_head_kv * f.page_size
                                + (int64_t) h * f.page_size + p % f.page_size;
                const uint16_t* row = (const uint16_t*) f.dev + (size_t) r * f.head_dim;
                const int64_t global_i_0 = r * f.head_dim + 0;
                const int64_t global_i_half = r * f.head_dim + f.n_rot / 2;
                const float a0 = 1.0f + (float) (global_i_0 % 7);
                const float b0 = 1.0f + (float) (global_i_half % 7);
                const int p_delta = d < 0 ? -d : d;
                const float s_sign = d < 0 ? -1.0f : 1.0f;
                const float c = f.cos_shadow[(size_t) p_delta * f.half + 0];
                const float s = s_sign * f.sin_shadow[(size_t) p_delta * f.half + 0];
                const float oa = a0 * c - b0 * s;
                const uint16_t want = fp32_to_fp16_bits(oa);
                if (row[0] != want) ++bad;
            }
        }
        mprotect(f.dev, f.total, PROT_NONE);
        check(bad == 0,
              "T7.3 non-uniform: every touched cell 0 matches its per-row delta's expected rotation");
    }

    // ---- T7.4 null-deltas guard: uniform=false, deltas=nullptr -> refuse ----
    {
        f.reset_k();
        fake_cuda::memcpy_calls = 0;
        auto st = f.make_state();
        const bool ok = rotate_qsa_tail_inplace(
            st, f.head_dim, f.n_rot, f.n_head_kv, (int) f.max_cells, f.page_size,
            /*edit_pos=*/4100, /*old_suffix_len=*/4, /*uniform=*/false, /*delta=*/0,
            /*deltas=*/nullptr, /*skip_rotation=*/false, f.pool_rows);
        check(!ok,
              "T7.4 null-deltas guard: uniform=false with deltas=nullptr -> refused (returns false)");
    }

    // ---- T7.5 pool-rows bounds: plan exceeds the pool -> refused (no write) ----
    {
        f.reset_k();
        fake_cuda::memcpy_calls = 0;
        auto st = f.make_state();
        st.n_slots = 1;  // tiny: 1*2*4 = 8 rows.  edit_pos+len would overflow this.
        const bool ok = rotate_qsa_tail_inplace(
            st, f.head_dim, f.n_rot, f.n_head_kv, (int) f.max_cells, f.page_size,
            /*edit_pos=*/4, /*old_suffix_len=*/4, /*uniform=*/true, /*delta=*/3,
            /*deltas=*/nullptr, /*skip_rotation=*/false, /*pool_rows=*/8);
        check(!ok,
              "T7.5 bounds: plan exceeding pool_rows is refused (no write, no SEGV)");
    }

    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
