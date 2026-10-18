// src/prefill/rp_moe.cu - see include/strata/prefill/rp_moe.hpp.  The kernels are mx-llama.cpp's (src/prefill/rp/,
// MIT); this file is the glue: the repack of a gathered expert, the routing tables, the launch.
#include "strata/prefill/rp_moe.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(STRATA_HIP_GFX906)
#include "rp/repack-kernels.cuh"
#endif

namespace strata::prefill::rp {

#if defined(STRATA_HIP_GFX906)

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill rp: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One thread per 32-value block: its 16 nibble bytes to the de-aliased nibble row, its f16 d to the scale plane
// (mx's repack_iq4nl_kernel, plus rows [row0, row0 + nrows) of a matrix and the row padding zeroed).
__global__ void repack_iq4nl_kernel(const uint8_t* __restrict__ src, size_t src_stride, uint8_t* __restrict__ dst,
                                    size_t dst_stride, int64_t rows, int64_t nb, int64_t qs_str, int64_t row0,
                                    int64_t nrows, int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % nb, r = (i / nb) % nrows, e = i / (nb * nrows);
        const int64_t row = row0 + r;
        const uint8_t* sb = src + (size_t) e * src_stride + (size_t) (r * nb + blk) * 18;
        uint8_t* m = dst + (size_t) e * dst_stride;
        // the source block is 2-byte aligned (18 B blocks): halfword loads, one 16-byte store
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(sb);
        uint4 q;
        q.x = (uint32_t) s16[1] | ((uint32_t) s16[2] << 16);
        q.y = (uint32_t) s16[3] | ((uint32_t) s16[4] << 16);
        q.z = (uint32_t) s16[5] | ((uint32_t) s16[6] << 16);
        q.w = (uint32_t) s16[7] | ((uint32_t) s16[8] << 16);
        *reinterpret_cast<uint4*>(m + (size_t) row * qs_str + (size_t) blk * 16) = q;
        reinterpret_cast<uint16_t*>(m + (size_t) rows * qs_str)[row * nb + blk] = s16[0];
        if (blk == 0)
            for (int64_t p = nb * 16; p < qs_str; p += 16)
                *reinterpret_cast<uint4*>(m + (size_t) row * qs_str + (size_t) p) = make_uint4(0, 0, 0, 0);
    }
}

struct GroupBlobs {
    const uint8_t* blob[mmq::kGatherGroupMax];
};

// The group gather: blockIdx.y = expert (from `first`), one thread per 32-value block of its gate, up and down rows,
// each into its repacked matrix (the same stores as repack_iq4nl_kernel).
__global__ void gather_group_iq4nl_kernel(GroupBlobs gb, int first, int64_t up_off, int64_t down_off, int64_t n_embd,
                                          int64_t n_ff, uint8_t* __restrict__ gu_dst, size_t gu_stride,
                                          uint8_t* __restrict__ d_dst, size_t d_stride) {
    const int q = first + (int) blockIdx.y;
    const uint8_t* blob = gb.blob[q];
    const int64_t nb_gu = n_embd / 32, nb_d = n_ff / 32;
    const int64_t n_gu = 2 * n_ff * nb_gu, total = n_gu + n_embd * nb_d;
    const int64_t qs_gu = repack_qs_row_stride(GGML_TYPE_IQ4_NL, n_embd), qs_d = repack_qs_row_stride(GGML_TYPE_IQ4_NL, n_ff);
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const uint8_t* sb;
        uint8_t* m;
        int64_t row, blk, nb, qs_str, rows;
        if (i < n_gu) {   // gate rows [0, n_ff) from +0, up rows [n_ff, 2 n_ff) from +up_off
            row = i / nb_gu; blk = i % nb_gu; nb = nb_gu; qs_str = qs_gu; rows = 2 * n_ff;
            sb = row < n_ff ? blob + (size_t) (row * nb_gu + blk) * 18
                            : blob + up_off + (size_t) ((row - n_ff) * nb_gu + blk) * 18;
            m = gu_dst + (size_t) q * gu_stride;
        } else {
            const int64_t j = i - n_gu;
            row = j / nb_d; blk = j % nb_d; nb = nb_d; qs_str = qs_d; rows = n_embd;
            sb = blob + down_off + (size_t) (row * nb_d + blk) * 18;
            m = d_dst + (size_t) q * d_stride;
        }
        const uint16_t* s16 = reinterpret_cast<const uint16_t*>(sb);
        uint4 v;
        v.x = (uint32_t) s16[1] | ((uint32_t) s16[2] << 16);
        v.y = (uint32_t) s16[3] | ((uint32_t) s16[4] << 16);
        v.z = (uint32_t) s16[5] | ((uint32_t) s16[6] << 16);
        v.w = (uint32_t) s16[7] | ((uint32_t) s16[8] << 16);
        *reinterpret_cast<uint4*>(m + (size_t) row * qs_str + (size_t) blk * 16) = v;
        reinterpret_cast<uint16_t*>(m + (size_t) rows * qs_str)[row * nb + blk] = s16[0];
        if (blk == 0)
            for (int64_t p = nb * 16; p < qs_str; p += 16)
                *reinterpret_cast<uint4*>(m + (size_t) row * qs_str + (size_t) p) = make_uint4(0, 0, 0, 0);
    }
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = (int32_t) i;
}

void launch_repack(const void* src, size_t src_stride, void* dst, size_t dst_stride, int64_t rows, int64_t cols,
                   int64_t row0, int64_t nrows, int n, cudaStream_t s) {
    const int64_t nb = cols / 32, total = (int64_t) n * nrows * nb;
    if (total <= 0) return;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_IQ4_NL, cols);
    const unsigned grid = (unsigned) std::min<int64_t>((total + 255) / 256, 4096);
    repack_iq4nl_kernel<<<grid, 256, 0, s>>>((const uint8_t*) src, src_stride, (uint8_t*) dst, dst_stride, rows, nb,
                                             qs_str, row0, nrows, total);
    ck(cudaGetLastError(), "repack_iq4nl");
}

// STRATA_PF_RP_W=32 | 64 forces a tile width; default: mx's rule (the 64-wide tile only once every expert fills it twice)
int tile_width(int64_t max_rows) {
    static const int forced = [] { const char* v = std::getenv("STRATA_PF_RP_W"); return v ? std::atoi(v) : 0; }();
    if (forced == 32 || forced == 64) return forced;
    return max_rows < 2 * 64 * MMQ_RP_Q8_TN ? 32 : 64;
}

}  // namespace

bool built() { return true; }

int rp_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_PF_RP"); return v ? std::atoi(v) : 0; }();
    return m;
}
bool enabled() { return rp_mode() == 1; }
bool sized() { return rp_mode() == 1 || rp_mode() == 2; }
bool dense_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_DENSE_RP"); return v && v[0] == '1'; }();
    return on;
}

bool supported(int t, int64_t w_rows, int64_t w_cols) {
    return t == GGML_TYPE_IQ4_NL && w_cols % 128 == 0 && w_rows % MMQ_RP_Q8_BM == 0;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) { return repack_gcn_nbytes((ggml_type) t, cols, rows); }

void repack_iq4nl(const void* src, size_t src_stride, void* dst, size_t dst_stride, int64_t rows, int64_t cols, int n,
                  void* stream) {
    launch_repack(src, src_stride, dst, dst_stride, rows, cols, 0, rows, n, (cudaStream_t) stream);
}

void repack_iq4nl_rows(const void* src, void* dst, int64_t rows, int64_t cols, int64_t row0, int64_t nrows,
                       void* stream) {
    launch_repack(src, 0, dst, 0, rows, cols, row0, nrows, 1, (cudaStream_t) stream);
}

void gather_group_iq4nl(const mmq::GatherGroup& g, size_t up_off, size_t down_off, int64_t n_embd, int64_t n_ff,
                        void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (g.first < 0 || g.n <= g.first || g.n > mmq::kGatherGroupMax) return;
    GroupBlobs gb{};
    for (int q = g.first; q < g.n; ++q) gb.blob[q] = g.blob[q];
    const int64_t total = 2 * n_ff * (n_embd / 32) + n_embd * (n_ff / 32);
    const unsigned gx = (unsigned) std::min<int64_t>((total + 255) / 256, 1024);
    gather_group_iq4nl_kernel<<<dim3(gx, (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        gb, g.first, (int64_t) up_off, (int64_t) down_off, n_embd, n_ff, (uint8_t*) gu_dst, gu_stride, (uint8_t*) d_dst,
        d_stride);
    ck(cudaGetLastError(), "gather_group_iq4nl");
}

Context::~Context() {
    if (iota_) cudaFree(iota_);
    if (tables_) cudaFree(tables_);
}

void Context::run(const mmq::Product& p, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (!supported(p.type, p.w_rows, p.w_cols)) {
        std::fprintf(stderr, "prefill rp: type %d [%lld x %lld] is not covered\n", p.type, (long long) p.w_rows,
                     (long long) p.w_cols);
        std::exit(1);
    }
    const int bw = tile_width(p.max_rows);
    const int64_t bn = bw == 32 ? 32 : 64 * MMQ_RP_Q8_TN;
    const int64_t max_tiles = (int64_t) p.n * ((p.max_rows + bn - 1) / bn);   // >= the tiles repack_tile_off writes
    if (p.total_rows > iota_rows_) {   // the identity source map (cudaFree waits for the device first)
        if (iota_) ck(cudaFree(iota_), "free");
        ck(cudaMalloc((void**) &iota_, (size_t) p.total_rows * 4), "iota");
        iota_kernel<<<(unsigned) std::min<int64_t>((p.total_rows + 255) / 256, 4096), 256, 0, s>>>(iota_, p.total_rows);
        iota_rows_ = p.total_rows;
    }
    const size_t off_bytes = ((size_t) (p.n + 1) * 4 + 255) & ~(size_t) 255;
    const size_t need = off_bytes + (size_t) max_tiles * sizeof(repack_tile_meta);
    if (need > tables_bytes_) {
        if (tables_) ck(cudaFree(tables_), "free");
        ck(cudaMalloc(&tables_, need), "tables");
        tables_bytes_ = need;
    }
    const int32_t* ids_src1 = iota_;
    int32_t* tile_off = (int32_t*) tables_;
    repack_tile_meta* tile_meta = (repack_tile_meta*) ((char*) tables_ + off_bytes);
    const dim3 grid((unsigned) ((p.w_rows + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM), (unsigned) max_tiles, 1);
    const uint8_t* w = (const uint8_t*) p.w;
    const block_q8_1* xq = (const block_q8_1*) p.xq;
    if (bw == 32) {
        repack_tile_off<32><<<1, 1, 0, s>>>(p.bounds, tile_off, tile_meta, p.n);
        mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_IQ4_NL>
            <<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, s>>>(
                w, xq, p.dst, (uint32_t) p.w_cols, (uint32_t) p.w_rows, (uint32_t) p.total_rows, ids_src1, p.ids,
                p.bounds, tile_off, tile_meta, (uint32_t) p.n, p.expert_bytes, (uint32_t) p.ld_dst);
    } else {
        repack_tile_off<64 * MMQ_RP_Q8_TN><<<1, 1, 0, s>>>(p.bounds, tile_off, tile_meta, p.n);
        mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_IQ4_NL>
            <<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, s>>>(
                w, xq, p.dst, (uint32_t) p.w_cols, (uint32_t) p.w_rows, (uint32_t) p.total_rows, ids_src1, p.ids,
                p.bounds, tile_off, tile_meta, (uint32_t) p.n, p.expert_bytes, (uint32_t) p.ld_dst);
    }
    ck(cudaGetLastError(), "rp gemm");
}

#else   // not gfx906: no repacked path

bool built() { return false; }
bool enabled() { return false; }
bool sized() { return false; }
bool dense_enabled() { return false; }
bool supported(int, int64_t, int64_t) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
void repack_iq4nl(const void*, size_t, void*, size_t, int64_t, int64_t, int, void*) {}
void repack_iq4nl_rows(const void*, void*, int64_t, int64_t, int64_t, int64_t, void*) {}
void gather_group_iq4nl(const mmq::GatherGroup&, size_t, size_t, int64_t, int64_t, void*, size_t, void*, size_t, void*) {}
Context::~Context() {}
void Context::run(const mmq::Product&, void*) {}

#endif

}  // namespace strata::prefill::rp
