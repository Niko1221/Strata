// src/prefill/rp_moe.cu - see include/strata/prefill/rp_moe.hpp.  The kernels are mx-llama.cpp's (src/prefill/rp/,
// MIT); this file is the glue: the repack of a gathered expert, the routing tables, the launch.
#include "strata/prefill/rp_moe.hpp"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#if defined(STRATA_HIP_GFX906)
#include "rp/repack-kernels.cuh"
#include "rp/repack-convert.cuh"
#include "rp/repack-iq4xs.cuh"
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
__global__ void repack_iq4nl_rows_kernel(const uint8_t* __restrict__ src, size_t src_stride, uint8_t* __restrict__ dst,
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
    // a part whose destination is null is not written (the caller gathers it as GGUF blocks for MMQ)
    const int64_t n_gu = gu_dst ? 2 * n_ff * nb_gu : 0, total = n_gu + (d_dst ? n_embd * nb_d : 0);
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
    repack_iq4nl_rows_kernel<<<grid, 256, 0, s>>>((const uint8_t*) src, src_stride, (uint8_t*) dst, dst_stride, rows, nb,
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

bool dense_supported(int t, int64_t w_rows, int64_t w_cols) {
    if (w_rows % MMQ_RP_Q8_BM != 0) return false;
    switch (t) {
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_Q8_0: return w_cols % 128 == 0;
        // Q4_K is left to llama.cpp's MMQ: dense_mmq_bench (T 4096, [6144 -> 2560]) measured the repacked GEMM 4% slower
        // (9.11 against 8.73 ms); Q6_K +19-26% and Q5_K +6% faster than MMQ
        case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_K: case GGML_TYPE_IQ4_XS: return w_cols % 256 == 0;
        default: return false;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    if (t == GGML_TYPE_IQ4_XS) return rp_iq4xs_nbytes(cols, rows);   // Strata's extension (rp/repack-iq4xs.cuh)
    return repack_gcn_nbytes((ggml_type) t, cols, rows);
}

void repack(int t, const void* src_v, size_t src_stride, void* dst_v, size_t dst_stride, int64_t ne1, int64_t ne0, int n,
            void* stream) {
    // mx-llama.cpp's set_tensor launch (buffer.cu), with ne2 = n matrices
    const cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* src = (const uint8_t*) src_v;
    uint8_t* dst = (uint8_t*) dst_v;
    const int64_t n_blocks = ne0 / 32,
                  qs_str = t == GGML_TYPE_IQ4_XS ? repack_qs_stride(ne0 / 2) : repack_qs_row_stride((ggml_type) t, ne0);
    const int64_t n_out = (int64_t) n * ne1 * n_blocks;
    const int block = 256, grid = (int) std::min<int64_t>((n_out + block - 1) / block, 65535);
    const int64_t ss = (int64_t) src_stride, ds = (int64_t) dst_stride;
    switch (t) {
        case GGML_TYPE_Q8_0:
            repack_q8_0_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, ne1 * qs_str, ss, ds, n_out);
            break;
        case GGML_TYPE_IQ4_NL:
            ::repack_iq4nl_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, ne1 * qs_str, ss, ds, n_out);
            break;
        case GGML_TYPE_Q6_K: {
            const int64_t nds = (n_blocks + 7) / 8, hoff = ne1 * qs_str, soff = hoff + ne1 * n_blocks * 8,
                          doff = soff + ne1 * n_blocks * 2;
            repack_q6k_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, hoff, soff, doff, nds, ss, ds, n_out);
        } break;
        case GGML_TYPE_Q4_K: {
            const int64_t nsb = (n_blocks + 7) / 8, soff = ne1 * qs_str, doff = soff + ne1 * nsb * 12;
            repack_q4k_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, soff, doff, nsb, ss, ds, n_out);
        } break;
        case GGML_TYPE_Q5_K: {
            const int64_t nsb = (n_blocks + 7) / 8, hoff = ne1 * qs_str, soff = hoff + ne1 * n_blocks * 4,
                          doff = soff + ne1 * nsb * 12;
            repack_q5k_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, hoff, soff, doff, nsb, ss, ds, n_out);
        } break;
        case GGML_TYPE_IQ4_XS:
            repack_iq4xs_kernel<<<grid, block, 0, s>>>(src, dst, ne1, n_blocks, qs_str, ne1 * qs_str, ss, ds, n_out);
            break;
        default:
            std::fprintf(stderr, "prefill rp: no repack for type %d\n", t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "repack");
}

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
    const int64_t total = (gu_dst ? 2 * n_ff * (n_embd / 32) : 0) + (d_dst ? n_embd * (n_ff / 32) : 0);
    if (total == 0) return;
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
    if (!dense_supported(p.type, p.w_rows, p.w_cols)) {
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
    auto launch = [&](auto wt) {
        constexpr ggml_type WT = decltype(wt)::value;
        if (bw == 32) {
            repack_tile_off<32><<<1, 1, 0, s>>>(p.bounds, tile_off, tile_meta, p.n);
            mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, WT>
                <<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, s>>>(
                    w, xq, p.dst, (uint32_t) p.w_cols, (uint32_t) p.w_rows, (uint32_t) p.total_rows, ids_src1, p.ids,
                    p.bounds, tile_off, tile_meta, (uint32_t) p.n, p.expert_bytes, (uint32_t) p.ld_dst);
        } else {
            repack_tile_off<64 * MMQ_RP_Q8_TN><<<1, 1, 0, s>>>(p.bounds, tile_off, tile_meta, p.n);
            mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, WT>
                <<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, s>>>(
                    w, xq, p.dst, (uint32_t) p.w_cols, (uint32_t) p.w_rows, (uint32_t) p.total_rows, ids_src1, p.ids,
                    p.bounds, tile_off, tile_meta, (uint32_t) p.n, p.expert_bytes, (uint32_t) p.ld_dst);
        }
    };
    switch (p.type) {
        case GGML_TYPE_IQ4_NL: launch(std::integral_constant<ggml_type, GGML_TYPE_IQ4_NL>{}); break;
        case GGML_TYPE_Q8_0: launch(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{}); break;
        case GGML_TYPE_Q6_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q6_K>{}); break;
        case GGML_TYPE_Q4_K: launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{}); break;
        case GGML_TYPE_IQ4_XS: launch(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{}); break;
        default: launch(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{}); break;
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
bool dense_supported(int, int64_t, int64_t) { return false; }
void repack(int, const void*, size_t, void*, size_t, int64_t, int64_t, int, void*) {}
void gather_group_iq4nl(const mmq::GatherGroup&, size_t, size_t, int64_t, int64_t, void*, size_t, void*, size_t, void*) {}
Context::~Context() {}
void Context::run(const mmq::Product&, void*) {}

#endif

}  // namespace strata::prefill::rp
