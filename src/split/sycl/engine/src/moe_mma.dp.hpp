// The prompt path's routed experts on the tensor cores, as llama.cpp's CUDA backend runs a mul_mat_id of more than
// MMQ_DP4A_MAX_BATCH_SIZE rows on a V100: each expert's weights dequantized to f16 (the f32 product d * q - m rounded
// once), the activations as f16, the products f16 x f16 with fp32 accumulation (llama.cpp's cuBLAS call accumulates
// in f16). A block of 4 warps takes one BigJob (one expert, up to PF_JOB rows) and MM_BM of the expert's rows: per
// MM_KC columns the weight tile is dequantized into shared memory next to the job's activation rows, and each warp runs
// a 32-row x 32-token corner of the output on HMMA.884 (mma_884: each quad-pair 2 x 2 tiles of 8 x 8, its operands
// 16-byte loads laid out without bank conflicts; shared-memory bandwidth is what limits these kernels on a V100). The
// next step's raw weights and
// activations are fetched into registers while the tensor cores run the current one. On random experts of the real
// shapes (tools/moe_mma_bench, 4,096 tokens) this takes gate|up 3.4 ms and down 2.2 ms per layer and card, against
// 4.8 and 4.0 ms for 8 warps of 16-row tiles at 128 columns.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
   // HIP: src/hip/mma.h (gfx12 WMMA)

namespace {

struct MoeTables {
    const int8_t* iq4;
    const uint8_t* mask;
    const uint8_t* signs;
    const uint32_t* iq3s;
    const uint32_t* iq3xxs;
    const uint64_t* iq2s;
};

// a k-step of MM_KC columns; a K-quant row takes MM_TPR threads (16 quant bytes, 32 weights each), so one pass of the
// block's threads fills MM_PASS rows; tile rows are padded to MM_LD halves (8: the fewest bank conflicts)
constexpr int MM_BM = 64, MM_KC = 64, MM_TPR = MM_KC / 32, MM_THREADS = 128, MM_PASS = MM_THREADS / MM_TPR, MM_LD = MM_KC + 8;
constexpr int MM_XV = PF_JOB * MM_KC / 8 / MM_THREADS;   // activation uint4s per thread and step, at most
// a K-quant tile's thread: its row and which half of the step's 64 columns (MM_TPR = 2). Eight threads in a row take 8
// rows at the same columns, so a quarter-warp's 16-byte stores hit 8 different bank groups (a 144-byte row stride); both
// halves of a row are still fetched by one warp (threads t and t + 8)
static_assert(MM_TPR == 2 && MM_THREADS == 128, "the K-quant thread layout");
__dpct_inline__ int mm_kq_row() {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
return (item_ct1.get_local_id(2) & 7) | ((item_ct1.get_local_id(2) >> 4) << 3); }
__dpct_inline__ int mm_kq_part() {
    return (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                2) >>
            3) &
               1;
}

// get_scale_min_k4 on a K-quant block's first 16 bytes (scales[0..11] are hd.y, hd.z, hd.w), by shifts: a byte index
// into a register would put it on the stack
__dpct_inline__ void mm_scale_min_k4(int j, const sycl::uint4 &hd, int &d,
                                     int &m) {
    if (j < 4) {
        d = (hd.y() >> (8 * j)) & 63;
        m = (hd.z() >> (8 * j)) & 63;
    } else {
        const int s = 8 * (j - 4);
        d = ((hd.w() >> s) & 0xF) | (((hd.y() >> (s + 6)) & 3) << 4);
        m = ((hd.w() >> (s + 4)) & 0xF) | (((hd.z() >> (s + 6)) & 3) << 4);
    }
}
// the 4 bytes of w (each 0..255) as two half2 {b0, b1}, {b2, b3}: each byte under 0x64 is 1024 + b exactly; minus `bias`
__dpct_inline__ void mm_bytes_h2(uint32_t w, sycl::half2 bias, sycl::half2 &a,
                                 sycl::half2 &b) {
    uint32_t x = dpct::byte_level_permute(w, 0x64646464u, 0x4140),
             y = dpct::byte_level_permute(w, 0x64646464u, 0x4342);
    a = *reinterpret_cast<sycl::half2 *>(&x) - bias;
    b = *reinterpret_cast<sycl::half2 *>(&y) - bias;
}
__dpct_inline__ sycl::half2 mm_k1024() {
    return sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(0x6400));
}
// bit k of the low 4 bits of t to bit 4 of byte k
__dpct_inline__ uint32_t mm_spread4(uint32_t t) {
    return ((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) | ((t & 8u) << 25);
}
// 16 values (8 half2) to shared memory
__dpct_inline__ void mm_store8(sycl::half *dst, const sycl::half2 *h) {
    reinterpret_cast<sycl::uint4 *>(dst)[0] =
        *reinterpret_cast<const sycl::uint4 *>(h);
    reinterpret_cast<sycl::uint4 *>(dst)[1] =
        *reinterpret_cast<const sycl::uint4 *>(h + 4);
}

// One k-step of the weight tile, MM_PASS rows x kc columns from column k0: fetch() reads this thread's raw bytes,
// put() writes their f16 values into shared memory, row stride MM_LD. The quants become f16 exactly (the 0x6400
// trick) and one half fma applies the scale: for Q5_1 and Q8_0 that is llama.cpp's dequantize_block_* value (its f32
// d * q + m is exact, rounded once to f16); for the K-quants the sub-block scale d * sc and min dmin * m are f16 here,
// rounded where llama.cpp keeps them f32 (at most an ulp of f16 per weight).
// The K-quants: 4 threads per row, each 16 bytes of the step's 64 quant bytes (one byte = a weight of sub-block 2j and
// one of 2j + 1). The 32-weight blocks: a block per thread (MM_TPR per row; those past a short last step idle).
template <int TYPE> struct MmTile;

// the K-quants' two sub-blocks 2j and 2j + 1 of one thread: lo / hi nibbles of the 16 quant bytes (plus, Q5_K, the
// high bits), scaled, into columns (i / 2) * 64 + l0 and + 32
template <bool Q5>

__dpct_inline__ void mm_put_k(const sycl::uint4 &hd, const sycl::uint4 &qv,
                              const sycl::uint4 &hv, bool live, int k0,
                              sycl::half *sw) {
    const int r = mm_kq_row(), i = mm_kq_part();
    const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
    sycl::half2 lo[8], hi[8];
    if (live) {
        const float
            d = sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        (uint16_t)(hd.x() & 0xFFFF)))
                    .convert<float, sycl::rounding_mode::automatic>()[0],
            dmin = sycl::vec<sycl::half, 1>(
                       sycl::bit_cast<sycl::half, unsigned short>(
                           (uint16_t)(hd.x() >> 16)))
                       .convert<float, sycl::rounding_mode::automatic>()[0];
        int sc, m;
        mm_scale_min_k4(2 * j, hd, sc, m);
        const sycl::half2
            d1 = sycl::float2(d * sc)
                     .convert<sycl::half, sycl::rounding_mode::rte>(),
            nm1 = sycl::float2(-(dmin * m))
                      .convert<sycl::half, sycl::rounding_mode::rte>();
        mm_scale_min_k4(2 * j + 1, hd, sc, m);
        const sycl::half2
            d2 = sycl::float2(d * sc)
                     .convert<sycl::half, sycl::rounding_mode::rte>(),
            nm2 = sycl::float2(-(dmin * m))
                      .convert<sycl::half, sycl::rounding_mode::rte>();
        const uint32_t q[4] = {qv.x(), qv.y(), qv.z(), qv.w()};
        const uint32_t h[4] = {hv.x(), hv.y(), hv.z(), hv.w()};
        const sycl::half2 k = mm_k1024();
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t a = q[w] & 0x0F0F0F0Fu, b = (q[w] >> 4) & 0x0F0F0F0Fu;
            if (Q5) {
                a |= ((h[w] >> (2 * j)) & 0x01010101u) << 4;
                b |= ((h[w] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            sycl::half2 x0, x1;
            mm_bytes_h2(a, k, x0, x1);
            lo[2 * w] = sycl::fma(x0, d1, nm1);
            lo[2 * w + 1] = sycl::fma(x1, d1, nm1);
            mm_bytes_h2(b, k, x0, x1);
            hi[2 * w] = sycl::fma(x0, d2, nm2);
            hi[2 * w + 1] = sycl::fma(x1, d2, nm2);
        }
    } else {
#pragma unroll
        for (int l = 0; l < 8; ++l) lo[l] = hi[l] =
            sycl::float2(0.f).convert<sycl::half, sycl::rounding_mode::rte>();
    }
    sycl::half *dst = sw + r * MM_LD + (i / 2) * 64 + l0;
    mm_store8(dst, lo);
    mm_store8(dst + 32, hi);
}

template <> struct MmTile<GGML_TYPE_Q4_K> {
    sycl::uint4 hd, qv;
    bool live;
    __dpct_inline__ void fetch(const char *we, size_t row_a, int nrows,
                               int row0, int k0, int) {
        const int r = mm_kq_row(), i = mm_kq_part(), row = row0 + r;
        const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
        live = row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 144;
            hd = *reinterpret_cast<const sycl::uint4 *>(b);
            qv = *reinterpret_cast<const sycl::uint4 *>(b + 16 + 32 * j + l0);
        }
    }
    __dpct_inline__ void put(int k0, int, sycl::half *sw) const {
        mm_put_k<false>(hd, qv, qv, live, k0, sw);
    }
};
template <> struct MmTile<GGML_TYPE_Q5_K> {
    sycl::uint4 hd, hv, qv;
    bool live;
    __dpct_inline__ void fetch(const char *we, size_t row_a, int nrows,
                               int row0, int k0, int) {
        const int r = mm_kq_row(), i = mm_kq_part(), row = row0 + r;
        const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
        live = row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 176;
            hd = *reinterpret_cast<const sycl::uint4 *>(b);
            hv = *reinterpret_cast<const sycl::uint4 *>(b + 16 + l0);
            qv = *reinterpret_cast<const sycl::uint4 *>(b + 48 + 32 * j + l0);
        }
    }
    __dpct_inline__ void put(int k0, int, sycl::half *sw) const {
        mm_put_k<true>(hd, qv, hv, live, k0, sw);
    }
};
template <> struct MmTile<GGML_TYPE_Q5_1> {
    sycl::uint2 v[3];
    bool live, item;
    __dpct_inline__ void fetch(const char *we, size_t row_a, int nrows,
                               int row0, int k0, int kc) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int r = item_ct1.get_local_id(2) / MM_TPR,
                  u = item_ct1.get_local_id(2) % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const sycl::uint2 *b = reinterpret_cast<const sycl::uint2 *>(
                (const uint8_t *)we + (size_t)row * row_a +
                (size_t)(k0 / 32 + u) * 24);
            v[0] = b[0];
            v[1] = b[1];
            v[2] = b[2];
        }
    }
    // byte l of the 16 quant bytes: value l (low nibble, high bit qh bit l) and value l + 16 (high nibble, qh bit l + 16)
    __dpct_inline__ void put(int, int kc, sycl::half *sw) const {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        if (!item) return;
        const int r = item_ct1.get_local_id(2) / MM_TPR,
                  u = item_ct1.get_local_id(2) % MM_TPR;
        sycl::half2 lo[8], hi[8];
        if (live) {
            const sycl::half2 d = sycl::half2(
                                  sycl::bit_cast<sycl::half, unsigned short>(
                                      (uint16_t)(v[0].x() & 0xFFFF))),
                              m = sycl::half2(
                                  sycl::bit_cast<sycl::half, unsigned short>(
                                      (uint16_t)(v[0].x() >> 16)));
            const uint32_t qh = v[0].y(),
                           q[4] = {v[1].x(), v[1].y(), v[2].x(), v[2].y()};
            const sycl::half2 k = mm_k1024();
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t a = (q[w] & 0x0F0F0F0Fu) | mm_spread4(qh >> (4 * w));
                const uint32_t b = ((q[w] >> 4) & 0x0F0F0F0Fu) | mm_spread4(qh >> (16 + 4 * w));
                sycl::half2 x0, x1;
                mm_bytes_h2(a, k, x0, x1);
                lo[2 * w] = sycl::fma(x0, d, m);
                lo[2 * w + 1] = sycl::fma(x1, d, m);
                mm_bytes_h2(b, k, x0, x1);
                hi[2 * w] = sycl::fma(x0, d, m);
                hi[2 * w + 1] = sycl::fma(x1, d, m);
            }
        } else {
#pragma unroll
            for (int l = 0; l < 8; ++l) lo[l] = hi[l] =
                sycl::float2(0.f)
                    .convert<sycl::half, sycl::rounding_mode::rte>();
        }
        mm_store8(sw + r * MM_LD + u * 32, lo);
        mm_store8(sw + r * MM_LD + u * 32 + 16, hi);
    }
};
template <> struct MmTile<GGML_TYPE_Q8_0> {
    uint16_t v[17];
    bool live, item;
    __dpct_inline__ void fetch(const char *we, size_t row_a, int nrows,
                               int row0, int k0, int kc) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int r = item_ct1.get_local_id(2) / MM_TPR,
                  u = item_ct1.get_local_id(2) % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const uint16_t* b = reinterpret_cast<const uint16_t*>((const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 32 + u) * 34);
#pragma unroll
            for (int l = 0; l < 17; ++l) v[l] = b[l];
        }
    }
    // the signed quants offset by 128 (bytes 0..255), so the f16 is 1152 + q: minus 1152, times d
    __dpct_inline__ void put(int, int kc, sycl::half *sw) const {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        if (!item) return;
        const int r = item_ct1.get_local_id(2) / MM_TPR,
                  u = item_ct1.get_local_id(2) % MM_TPR;
        sycl::half2 y[16];
        if (live) {
            const sycl::half2 d =
                sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(v[0]));
            const sycl::half2 k = sycl::half2(
                sycl::bit_cast<sycl::half, unsigned short>(0x6480)); // 1152
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t a = ((uint32_t) v[1 + 2 * w] | ((uint32_t) v[2 + 2 * w] << 16)) ^ 0x80808080u;
                sycl::half2 x0, x1;
                mm_bytes_h2(a, k, x0, x1);
                y[2 * w] = x0 * d;
                y[2 * w + 1] = x1 * d;
            }
        } else {
#pragma unroll
            for (int l = 0; l < 16; ++l)
                y[l] = sycl::float2(0.f)
                           .convert<sycl::half, sycl::rounding_mode::rte>();
        }
        mm_store8(sw + r * MM_LD + u * 32, y);
        mm_store8(sw + r * MM_LD + u * 32 + 16, y + 8);
    }
};

// the job's activation rows for one k-step: rows past n (up to the token tiles' 16 * nt) are zeros. A thread's rows and
// columns are the same at every step (element e = threadIdx.x + i * MM_THREADS is row e / MM_V8, 8 columns e % MM_V8)
constexpr int MM_V8 = MM_KC / 8;
struct MmX {
    sycl::uint4 v[MM_XV];
    int y[MM_XV];   // the row's token, -1 past the job's rows, -2 past its token tiles
    __dpct_inline__ void rows(const BigJob &jb, int nt) {
#pragma unroll
        for (int i = 0; i < MM_XV; ++i) {
            const int j = (sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                               .get_local_id(2) +
                           i * MM_THREADS) /
                          MM_V8;
            y[i] = j < jb.n ? jb.y[j] : j < nt * 16 ? -1 : -2;
        }
    }
    __dpct_inline__ void fetch(const sycl::half *x, int ldx, int k0, int kc) {
        const int c =
            sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
                2) %
            MM_V8 * 8; // MM_THREADS is a multiple of MM_V8
#pragma unroll
        for (int i = 0; i < MM_XV; ++i) {
            v[i] = sycl::uint4(0, 0, 0, 0);
            if (y[i] >= 0 && c < kc) v[i] =
                *reinterpret_cast<const sycl::uint4 *>(x + (size_t)y[i] * ldx +
                                                       k0 + c);
        }
    }
    __dpct_inline__ void put(int kc, sycl::half *sx) const {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int c = item_ct1.get_local_id(2) % MM_V8 * 8;
#pragma unroll
        for (int i = 0; i < MM_XV; ++i)
            if (y[i] >= -1 && c < kc) *
                reinterpret_cast<sycl::uint4 *>(
                    sx +
                    (item_ct1.get_local_id(2) + i * MM_THREADS) / MM_V8 *
                        MM_LD +
                    c) = v[i];
    }
};

#if FNX_HIP
// gfx12 (RDNA4): v_wmma_f32_16x16x16_f16, wave32. A is 16 weight rows x 16 k (lane l: row l & 15, k 8 * (l >> 4) + 0..7), B is 16 k x
// 16 tokens (lane l: token l & 15, the same k), the accumulator lane l holds token l & 15 and rows 8 * (l >> 4) + 0..7. f16 products,
// fp32 accumulation, as the HMMA.884 did.
// gfx11 (RDNA3, FNX_WMMA_GFX11, not run on a card yet; hip/selftest.h checks it): an operand is 16 k a lane, the same in lanes l and l + 16; the accumulator lane l holds
// rows 2 j + (l >> 4) (hip/mma.h)
typedef nvcuda::wmma::fnx_hab mm_h8;   // an A or B operand: 8 (gfx12) or 16 (gfx11) halves
typedef float mm_f8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ mm_f8 mma_wmma12(const mm_h8& a, const mm_h8& b, const mm_f8& c) {
#if FNX_WMMA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    (void) a; (void) b;
    __builtin_trap();   // not a gfx12 device pass: the launcher never selects it
    return c;
#endif
}
#elif !FNX_SYCL
// d (each quad-pair's 8 x 8) += a (8 x 4, row-major: this lane's row, 4 halves in a0 a1) b (4 x 8, col-major: this
// lane's column), f16 products with f32 accumulation: Volta's HMMA.884
__dpct_inline__ void mma_884(float (&d)[8], unsigned a0, unsigned a1,
                             unsigned b0, unsigned b1) {
    
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
                   "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif

#if FNX_ARCH >= 80 && !FNX_HIP
// d (16 x 8) += a (16 x 16, row-major) b (16 x 8, col-major): Ampere's HMMA.16816, f16 products with f32 accumulation. Lane (g = lane / 4,
// t = lane % 4) holds a0 = (row g, k 2t..2t+1), a1 = (row g + 8, k 2t..), a2 = (row g, k 2t + 8..), a3 = (row g + 8, k 2t + 8..);
// b0 = (k 2t..2t+1, column g), b1 = (k 2t + 8.., column g); d0 d1 = (row g, columns 2t, 2t + 1), d2 d3 = (row g + 8, same columns)
__dpct_inline__ void mma_16816(float (&d)[4], uint32_t a0, uint32_t a1,
                               uint32_t a2, uint32_t a3, uint32_t b0,
                               uint32_t b1) {
    {
        volatile void *d_mat_frag_ct1[4] = {&d[0], &d[1], &d[2], &d[3]};
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(a0, a1, a2, a3);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(b0, b1);
        sycl::vec<float, 4> c_mat_frag_ct1(d[0], d[1], d[2], d[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1,
            &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
}
#endif

// the output staging's row stride in floats (8 past the tile: fewer conflicts than none or 16)
template <int BM>
constexpr int MM_SLD = BM + 8;
// the block's shared memory: the weight and activation tiles, later the output staging over them
template <int BM>
constexpr size_t mm_smem() {
    return std::max(sizeof(sycl::half) * (BM + PF_JOB) * MM_LD,
                    PF_JOB *MM_SLD<BM> * sizeof(float));
}

// out[jb.out[j] * nrows + row] = sum_k W_e[row][k] * x[jb.y[j] * ldx + k] for the job's rows j, BM rows of the expert
// from row BM * blockIdx.x. Each warp holds a 32-row x (64 / WT)-token output (2 x TT WMMA tiles, so a fragment loaded
// from shared memory feeds TT or 2 products): warp w takes rows 32 (w % WR) and tokens (64 / WT) (w / WR). A job of at
// most TT token tiles runs on every warp anyway: the WT warps of a row group then take every WT-th 16 columns of each
// k-step, summed at the end.
template <int TYPE, int BM>

void k_moe_mma(MoeW w, const BigJob *__restrict__ jobs,
               const int *__restrict__ njobs, const sycl::half *__restrict__ x,
               int ldx, float *__restrict__ out, uint8_t *dpct_local,
               BigJob &jb, MoeTables tables) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    constexpr int MM_NT = PF_JOB / 16;   // token tiles of a job
    constexpr int WR = BM / 32, WT = MM_THREADS / 32 / WR, TT = MM_NT / WT, NP = BM / MM_PASS, LO = 16 * TT;
    static_assert(WR * WT == MM_THREADS / 32 && TT * WT == MM_NT && NP * MM_PASS == BM, "warp layout");
#if FNX_ARCH < 80
    static_assert(PF_JOB == 64, "the m8n8k4 path is laid out for 64-row jobs");
#endif
    const int ji = item_ct1.get_group(1);
    if (ji >= *njobs) return;

    auto smem = (sycl::half *)dpct_local; // the weight tile (BM rows), then the
                                          // activation rows (PF_JOB)
    sycl::half *sw = smem;
    sycl::half *sx = smem + BM * MM_LD;
    load_bigjob(jb, jobs + ji);
    const int n = jb.n, nt = (n + 15) / 16,
              warp = item_ct1.get_local_id(2) >> 5,
              row0 = item_ct1.get_group(2) * BM;
    const int rg = warp % WR, tg = warp / WR;
    const bool split = WT > 1 && nt <= TT;
    const int tb = split ? 0 : tg * TT;   // the warp's first token tile
#if FNX_HIP
    // the warp's 32 rows x 16 TT tokens: 2 x TT WMMA tiles of 16 x 16 (lane l loads weight row / token l & 15 of a tile, 8 k at
    // 8 * (l >> 4))
    const int lane = threadIdx.x & 31, ln = lane & 15, lh = lane >> 4;
    const int kq = FNX_WMMA_GFX11 ? 0 : 8 * lh;                              // gfx12: this lane's 8 k of 16; gfx11: all 16
    const __half* arow = sw + (rg * 32 + ln) * MM_LD + kq;                   // + 16 MM_LD for tile row 1
    const __half* bcol = sx + (tb * 16 + ln) * MM_LD + kq;                   // + 16 MM_LD per token tile
    mm_f8 acc[2][TT];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < TT; ++b)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[a][b][j] = 0.f;
#elif FNX_ARCH >= 80
    // the warp's 32 rows x 16 TT tokens on HMMA.16816 (mma_16816): 2 row tiles of 16 x 2 TT token tiles of 8; lane (g, t4) loads the
    // 32-bit operand pairs of the fragment layout above from the weight tile (rows) and the activation tile (tokens) in shared memory
    const int lane = item_ct1.get_local_id(2) & 31, g = lane >> 2,
              t4 = lane & 3;
    const sycl::half *arow = sw + (rg * 32 + g) * MM_LD + 2 * t4;
    const sycl::half *bcol = sx + (tb * 16 + g) * MM_LD + 2 * t4;
    float acc[2][2 * TT][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2 * TT; ++b)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[a][b][j] = 0.f;
#else
    // the warp's 32 rows x 16 TT tokens on HMMA.884 (mma_884): quad-pair q (lanes 4q..4q+3, 4q+16..4q+19) takes rows
    // 16 (q >> 1) and tokens 8 TT (q & 1) of them, as 2 x TT tiles of 8 x 8; this lane loads row / token `rl` of a tile
    const int lane = threadIdx.x & 31, q = (lane >> 2) & 3, rl = (lane & 3) + ((lane >> 4) & 1) * 4;
    const __half* arow = sw + (rg * 32 + (q >> 1) * 16 + rl) * MM_LD;            // + 8 MM_LD for tile row 1
    // quad-pair 1's lanes take token rl ^ 4 of a tile: their loads share wavefronts with quad-pair 0's, whose rows are rl,
    // and rows 8 apart share banks (a 144-byte stride); its output columns are swizzled back below
    const int sw4 = (q & 1) << 2;
    const __half* bcol = sx + (tb * 16 + (q & 1) * 8 * TT + (rl ^ sw4)) * MM_LD;   // + 8 MM_LD per token tile
    float acc[2][TT][8];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < TT; ++b)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[a][b][j] = 0.f;
#endif
    const char* we = moe_expert(w, jb.w);
    MmTile<TYPE> wt[NP];
    MmX xt;
    xt.rows(jb, nt);
#pragma unroll
    for (int p = 0; p < NP; ++p) wt[p].fetch(
        we, w.row_a, w.nrows, row0 + p * MM_PASS, 0, sycl::min(MM_KC, w.ncols));
    xt.fetch(x, ldx, 0, sycl::min(MM_KC, w.ncols));
    for (int k0 = 0; k0 < w.ncols; k0 += MM_KC) {
        const int kc = sycl::min(MM_KC, w.ncols - k0);
#pragma unroll
        
        for (int p = 0; p < NP; ++p) {
            auto* dst = sw + p * MM_PASS * MM_LD;
            if constexpr (TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_IQ4_NL ||
                          TYPE == GGML_TYPE_IQ4_XS)
                wt[p].put(k0, kc, dst, tables.iq4);
            else if constexpr (TYPE == GGML_TYPE_IQ3_S)
                wt[p].put(k0, kc, dst, tables.mask, tables.iq3s);
            else if constexpr (TYPE == GGML_TYPE_IQ3_XXS)
                wt[p].put(k0, kc, dst, tables.mask, tables.signs, tables.iq3xxs);
            else if constexpr (TYPE == GGML_TYPE_IQ2_S)
                wt[p].put(k0, kc, dst, tables.mask, tables.iq2s);
            else
                wt[p].put(k0, kc, dst);
        }
        xt.put(kc, sx);
        
        
        item_ct1.barrier();
        if (k0 + MM_KC < w.ncols) {
            const int kn = sycl::min(MM_KC, w.ncols - k0 - MM_KC);
#pragma unroll
            for (int p = 0; p < NP; ++p) wt[p].fetch(we, w.row_a, w.nrows, row0 + p * MM_PASS, k0 + MM_KC, kn);
            xt.fetch(x, ldx, k0 + MM_KC, kn);
        }
        if (tb < nt) {
            const int kt0 = split ? tg : 0, kts = split ? WT : 1;
#pragma unroll
            for (int i = 0; i < MM_KC / 16; ++i) {
                const int kt = kt0 + i * kts, kk = kt * 16;
                if (kt >= kc / 16) break;
#if FNX_HIP
                mm_h8 a[2], b[TT];
#pragma unroll
                for (int r = 0; r < 2; ++r) a[r] = nvcuda::wmma::fnx_ld_ab(arow + r * 16 * MM_LD + kk);
#pragma unroll
                for (int t = 0; t < TT; ++t) b[t] = nvcuda::wmma::fnx_ld_ab(bcol + t * 16 * MM_LD + kk);
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int t = 0; t < TT; ++t) acc[r][t] = mma_wmma12(a[r], b[t], acc[r][t]);
#elif FNX_ARCH >= 80
                uint32_t fa[2][4], fb[2 * TT][2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const sycl::half *ap = arow + r * 16 * MM_LD + kk;
                    fa[r][0] = *reinterpret_cast<const uint32_t*>(ap);
                    fa[r][1] = *reinterpret_cast<const uint32_t*>(ap + 8 * MM_LD);
                    fa[r][2] = *reinterpret_cast<const uint32_t*>(ap + 8);
                    fa[r][3] = *reinterpret_cast<const uint32_t*>(ap + 8 * MM_LD + 8);
                }
#pragma unroll
                for (int u = 0; u < 2 * TT; ++u) {
                    const sycl::half *bp = bcol + u * 8 * MM_LD + kk;
                    fb[u][0] = *reinterpret_cast<const uint32_t*>(bp);
                    fb[u][1] = *reinterpret_cast<const uint32_t*>(bp + 8);
                }
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int u = 0; u < 2 * TT; ++u) mma_16816(acc[r][u], fa[r][0], fa[r][1], fa[r][2], fa[r][3], fb[u][0], fb[u][1]);
#else
                // 16 k: a 16-byte load per tile row (8 k) twice; each 4 k of it is one HMMA's operand
                uint4 a[2][2], b[TT][2];
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int h = 0; h < 2; ++h) a[r][h] = *reinterpret_cast<const uint4*>(arow + r * 8 * MM_LD + kk + h * 8);
#pragma unroll
                for (int t = 0; t < TT; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) b[t][h] = *reinterpret_cast<const uint4*>(bcol + t * 8 * MM_LD + kk + h * 8);
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4)
#pragma unroll
                    for (int r = 0; r < 2; ++r)
#pragma unroll
                        for (int t = 0; t < TT; ++t) {
                            const uint4& A = a[r][k4 >> 1];
                            const uint4& B = b[t][k4 >> 1];
                            mma_884(acc[r][t], k4 & 1 ? A.z : A.x, k4 & 1 ? A.w : A.y, k4 & 1 ? B.z : B.x, k4 & 1 ? B.w : B.y);
                        }
#endif
            }
        }
        
        
        item_ct1.barrier();
    }
    // the output through shared memory over the tiles, token-major [token][row] with rows padded to MM_SLD<BM> floats,
    // so the read back below takes consecutive rows; a split's partial sums are one such [LO][row] block per tg
    float* so = reinterpret_cast<float*>(smem);
    constexpr int SLD = MM_SLD<BM>;
    static_assert(MM_LD % 8 == 0, "16-byte operand loads");
#if FNX_HIP
    // the accumulator lane l holds token ln and rows 8 lh + 0..7 of a tile (gfx12): eight consecutive floats of the staging's row; gfx11: rows 2 j + lh
    float* sob = so + (split ? tg * LO * SLD : tb * 16 * SLD) + rg * 32 + (FNX_WMMA_GFX11 ? lh : 8 * lh);
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            float* d = sob + (t * 16 + ln) * SLD + r * 16;
#pragma unroll
            for (int j = 0; j < 8; ++j) d[FNX_WMMA_GFX11 ? 2 * j : j] = acc[r][t][j];
        }
#elif FNX_ARCH >= 80
    // a lane's accumulators of a 16 x 8 tile: rows g and g + 8, columns 2 t4 and 2 t4 + 1 (mma_16816)
    float* sob = so + (split ? tg * LO * SLD : tb * 16 * SLD) + rg * 32;   // the warp's rows, its first token
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int u = 0; u < 2 * TT; ++u)
#pragma unroll
            for (int j = 0; j < 4; ++j) sob[(u * 8 + 2 * t4 + (j & 1)) * SLD + r * 16 + g + (j >> 1) * 8] = acc[r][u][j];
#else
    // a lane's accumulators in an 8 x 8 tile (measured, tools/m884_probe.cu): rows r0 and r0 + 2, columns cb, cb + 1,
    // cb + 4, cb + 5, in the order (r0,cb) (r0,cb+1) (r1,cb) (r1,cb+1) (r0,cb+4) (r0,cb+5) (r1,cb+4) (r1,cb+5)
    const int r0 = (lane & 1) + ((lane >> 4) & 1) * 4, cb = lane & 2;
    float* sob = so + (split ? tg * LO * SLD : tb * 16 * SLD) + rg * 32 + (q >> 1) * 16;   // the warp's rows, its first token
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = r * 8 + r0 + ((j >> 1) & 1) * 2;
                const int tok = (q & 1) * 8 * TT + t * 8 + ((cb + (j & 1) + ((j >> 2) & 1) * 4) ^ sw4);
                sob[tok * SLD + row] = acc[r][t][j];
            }
#endif
    
    item_ct1.barrier();
    for (int e = item_ct1.get_local_id(2); e < n * BM; e += MM_THREADS) {
        const int j = e / BM, r = e % BM, row = row0 + r;
        if (row >= w.nrows) continue;
        float v = so[j * SLD + r];
        if (split) {
#pragma unroll
            for (int g = 1; g < WT; ++g) v += so[g * LO * SLD + j * SLD + r];
        }
        out[(size_t) jb.out[j] * w.nrows + row] = v;
    }
}

// h = silu(gate) * up of the stacked gate|up rows, rounded to f16 (llama.cpp's swiglu output, converted for cuBLAS)
void k_swiglu_h16(const float *__restrict__ gu, int ffs, int P,
                  sycl::half *__restrict__ h) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (size_t e =
             item_ct1.get_group(2) * (size_t)item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         e < (size_t)P * ffs; e += (size_t)item_ct1.get_group_range(2) *
                                   item_ct1.get_local_range(2)) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = sycl::vec<float, 1>(silu_(g) * gu[r * 2 * ffs + ffs + i])
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

}  // namespace
