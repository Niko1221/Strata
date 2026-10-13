// opencl/probe/ocl_iqfmt.cpp - the four remaining Strata expert formats, on the MI50, via OpenCL.
//
//   type 21  IQ3_S  gu    110 B / 256 values   (10 layers)
//   type 22  IQ2_S  gu     82 B / 256 values   (20 layers)
//   type 23  IQ4_XS gu    136 B / 256 values   ( 1 layer)
//   type 42  Q2_0   down   18 B /  64 values   ( 9 layers)
//
// IQ3_XXS (gu) and IQ4_NL (down) already run here; with these four every layer of the iq3_s pack is
// covered, which is what makes the MI50 useful as a second expert tier rather than for 14 layers out of 48.
//
// One kernel body, four decodes: the row walk, the dp4a loop and the reduction are format-independent
// (all four are "decode a 32-value group into eight int32 of four signed bytes, dot with the matching
// q8_1 block"), so only the `#if IQ_FMT` block changes.  The reference data and the expected row dots
// come from iqfmt_ref.py, which checks its own layout two independent ways.
//
// The one byte-offset rule this file depends on, from iq_kernels.cu:
//     int get_int_b2(const void* x, int i) { const uint16_t* x16 = x; return x16[2*i] | (x16[2*i+1] << 16); }
// is a 32-bit load at byte offset 4*i - so every `iqs` in the CUDA source is a word index, not a byte
// or element index.  Reading `Split<22>::load` with iqs = 2c is what puts IQ2_S's grid bytes at qs[4c]
// and its sign bytes at qs[32+4c].
//
// Usage: ocl_iqfmt [repeat] [rep] [-1|0|1] [fmt21 fmt22 ...] [--sg]

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* err_name(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        default: return "CL_ERR";
    }
}

bool read_file(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const size_t got = std::fread(out.data(), 1, (size_t) n, f);
    std::fclose(f);
    return got == (size_t) n;
}

uint32_t rd_u32(const char* p, size_t off) {
    uint32_t v;
    std::memcpy(&v, p + off, 4);
    return v;
}
float rd_f32(const char* p, size_t off) {
    float v;
    std::memcpy(&v, p + off, 4);
    return v;
}

// ---- shared device helpers ---------------------------------------------------------------------------
//
// dp4x8 is mad24 x4 with sign-extended bytes: bit-exact against __dp4a and the fastest form this driver
// has (ocl_dot.cpp).  vcmpne4 is the explicit byte loop because the bit trick this driver generates for
// it loses bit 31 of the mask (ocl_iq3.cpp).

const char* kHelpers = R"CLC(
inline float f16_to_f(uint h) {
    const uint s = (h >> 15u) & 1u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 31u) return as_float((s << 31u) | 0x7F800000u | (m << 13u));
    if (e == 0u) {
        if (m == 0u) return as_float(s << 31u);
        const float v = (float) m * 5.9604644775390625e-08f;   // denormal
        return s ? -v : v;
    }
    return as_float((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

inline uint rd16(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8);
}
inline uint rd32(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8) | ((uint) p[o + 2u] << 16) | ((uint) p[o + 3u] << 24);
}

inline uint unpack_ksigns(uint v) {
    const uint p = (uint) popcount(v & 0xFFu) & 1u;
    return (v & 0xFFu) ^ (p << 7u);
}

inline int vcmpne4(int a, int b) {
    const uint t = as_uint(a) ^ as_uint(b);
    uint r = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (((t >> (8 * i)) & 0xFFu) != 0u) r |= 0xFFu << (8 * i);
    return as_int(r);
}

inline int vsub4(int a, int b) {
    const uint ua = as_uint(a), ub = as_uint(b);
    const uint kHigh = 0x80808080u;
    return as_int(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

inline int dp4x8(int a, int b, int c) {
    int s = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        s = mad24((a << (24 - 8 * i)) >> 24, (b << (24 - 8 * i)) >> 24, s);
    return s;
}

inline int pack4(int b0, int b1, int b2, int b3) {
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}
)CLC";

// ---- the kernel --------------------------------------------------------------------------------------
//
// IQ_WG lanes walk the calls of a row together; IQ_R rows share one pass over the activations, because at
// 2-3.5 bits per weight the activation traffic is several times the weight traffic.
//
// One call = 32 weights.  k counts calls: kbx = k / ipb is the weight block, m = k % ipb the sub-block
// inside it, and the q8_1 activation block is kbx*ipb + m - the activation blocks are numbered by
// 32-value position along the whole row, not from the start of the weight block.  Getting that wrong
// leaves every dot product uncorrelated while still producing plausible numbers.

std::string source(int fmt, int wg, int rows, bool subgroup) {
    const int ipb = (fmt == 42) ? 2 : 8;
    const int step = (fmt == 21 || fmt == 22) ? 2 : (fmt == 23 ? 4 : 1);
    const int row_vals = (fmt == 42) ? 640 : 2560;
    const int qk = (fmt == 42) ? 64 : 256;
    const int nchunk = (row_vals / qk) * ipb;

    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";
    s += "#define IQ_FMT " + std::to_string(fmt) + "\n";
    s += "#define IQ_WG " + std::to_string(wg) + "\n";
    s += "#define IQ_R " + std::to_string(rows) + "\n";
    s += "#define IQ_IPB " + std::to_string(ipb) + "\n";
    s += "#define IQ_NCHUNK " + std::to_string(nchunk) + "\n";
    if (subgroup) s += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n#define IQ_SG 1\n";
    else s += "#define IQ_LMEM 1\n";
    s += kHelpers;
    s += R"CLC(
// Decode one 32-value group into eight int32 of four signed bytes, ordered so that g[j] pairs with the
// j-th q8_1 word of this call's activation block.
inline void group_decode(__global const uchar* w, uint base, uint m, uint iqs,
                         __global const uint* t2, __global const uint* t3, __global const uchar* kv,
                         int g[8])
{
#if IQ_FMT == 21
    // block_iq3_s: d(2) qs[64] qh[8] signs[32] scales[4]
    const uint c = m;
    __global const uchar* qs = w + base + 2u + 8u * c;
    const uint qh = w[base + 2u + 64u + c];
    const uint sgb = rd32(w, base + 2u + 72u + 4u * c);
    #pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint i1 = (uint) qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100u);
        const uint i2 = (uint) qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100u);
        const uint sg = (sgb >> (8u * (uint) il)) & 0xFFu;
        const int m0 = vcmpne4((int) (((sg & 0x03u) << 7) | ((sg & 0x0Cu) << 21)), 0);
        const int m1 = vcmpne4((int) (((sg & 0x30u) << 3) | ((sg & 0xC0u) << 17)), 0);
        g[2 * il + 0] = vsub4((int) t3[i1] ^ m0, m0);
        g[2 * il + 1] = vsub4((int) t3[i2] ^ m1, m1);
    }
#elif IQ_FMT == 22
    // block_iq2_s: d(2) qs[64] qh[8] scales[8]; qs[0..32) are the 10-bit grid indices, qs[32..64) the signs
    const uint c = m;
    __global const uchar* qs = w + base + 2u + 4u * c;
    const uint qh = w[base + 2u + 64u + c];
    const uint sgb = rd32(w, base + 2u + 32u + 4u * c);
    #pragma unroll
    for (int il = 0; il < 4; ++il) {
        const uint idx = (uint) qs[il] | ((qh << (8 - 2 * il)) & 0x300u);
        const uint lo = t2[2u * idx + 0u];        // grid_pos[0]: the entry's low four bytes
        const uint hi = t2[2u * idx + 1u];        // grid_pos[1]
        const uint sg = (sgb >> (8u * (uint) il)) & 0xFFu;
        const int m0 = vcmpne4((int) (((sg & 0x03u) << 7) | ((sg & 0x0Cu) << 21)), 0);
        const int m1 = vcmpne4((int) (((sg & 0x30u) << 3) | ((sg & 0xC0u) << 17)), 0);
        g[2 * il + 0] = vsub4((int) lo ^ m0, m0);
        g[2 * il + 1] = vsub4((int) hi ^ m1, m1);
    }
#elif IQ_FMT == 23
    // block_iq4_xs: d(2) scales_h(2) scales_l[4] qs[128]; sub-block m is qs[16m .. 16m+16)
    __global const uchar* q4 = w + base + 8u + 16u * m;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint word = rd32(q4, (uint) (4 * j));
        g[j + 0] = pack4((int) (char) kv[word & 0x0Fu], (int) (char) kv[(word >> 8) & 0x0Fu],
                         (int) (char) kv[(word >> 16) & 0x0Fu], (int) (char) kv[(word >> 24) & 0x0Fu]);
        g[j + 4] = pack4((int) (char) kv[word >> 4 & 0x0Fu], (int) (char) kv[(word >> 12) & 0x0Fu],
                         (int) (char) kv[(word >> 20) & 0x0Fu], (int) (char) kv[word >> 28]);
    }
#else
    // block_q2_0: float d + qs[16], 64 two-bit values; one call is half the block
    __global const uchar* qs = w + base + 2u + 8u * iqs;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint q = rd16(qs, (uint) (2 * j));      // 8 two-bit codes
        g[2 * j + 0] = pack4((int) ((q >> 0) & 3u) - 1, (int) ((q >> 2) & 3u) - 1,
                             (int) ((q >> 4) & 3u) - 1, (int) ((q >> 6) & 3u) - 1);
        g[2 * j + 1] = pack4((int) ((q >> 8) & 3u) - 1, (int) ((q >> 10) & 3u) - 1,
                             (int) ((q >> 12) & 3u) - 1, (int) ((q >> 14) & 3u) - 1);
    }
#endif
}

// apply(): the per-format integer scale, and how the eight dp4a words combine.
inline float apply(__global const uchar* w, uint base, uint m, uint iqs, const int g[8], const uint xw[8],
                   float dw, float dx)
{
#if IQ_FMT == 21
    const int ls = 1 + 2 * (int) ((w[base + 2u + 104u + m / 2u] >> (4u * (m & 1u))) & 0x0Fu);
    int si = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);
    return dw * dx * (float) (si * ls);
#elif IQ_FMT == 22
    const uint sc = w[base + 2u + 72u + m];
    const int ls0 = (int) (sc & 0x0Fu), ls1 = (int) (sc >> 4);
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) s0 = dp4x8(g[j], (int) xw[j], s0);
    #pragma unroll
    for (int j = 4; j < 8; ++j) s1 = dp4x8(g[j], (int) xw[j], s1);
    const int sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
    return dw * dx * (float) sumi;
#elif IQ_FMT == 23
    const uint sh = rd16(w, base + 2u);
    const int ls = (int) (((w[base + 4u + m / 2u] >> (4u * (m & 1u))) & 0x0Fu) | (((sh >> (2u * m)) & 3u) << 4));
    int si = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        si = dp4x8(g[j], (int) xw[j], si);
        si = dp4x8(g[j + 4], (int) xw[j + 4], si);
    }
    return dw * dx * (float) (si * (ls - 32));
#else
    int si = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);
    return dw * dx * (float) si;      // dw already holds the block's float scale for Q2_0
#endif
}

__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iqfmt_row(__global const uchar* w, __global const uint* xint, __global const float* xf16,
               __global const uint* t2, __global const uint* t3, __global const uchar* kv,
               __global float* outv, const uint wrow_bytes, const uint n_rows
#ifdef IQ_LMEM
               , __local float* lred
#endif
               )
{
    const uint lane = get_local_id(0);
    const uint row0 = get_group_id(0) * (uint) IQ_R;

    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

    for (uint k = lane; k < (uint) IQ_NCHUNK; k += (uint) IQ_WG) {
#if IQ_FMT == 42
        const uint kbx = k >> 1, m = k & 1u, iqs = m;
#elif IQ_FMT == 23
        const uint kbx = k >> 3, m = k & 7u, iqs = 4u * m;
#else
        const uint kbx = k >> 3, m = k & 7u, iqs = 2u * m;
#endif
        const uint xb = kbx * (uint) IQ_IPB + m;          // the q8_1 block this call multiplies

        // 36 bytes per q8_1 block: fp16 d, fp16 sum, 32 int8.  Read once, feed all IQ_R rows.
        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = xint[xb * 9u + 1u + (uint) j];
        const float dx = xf16[xb * 2u];

        #pragma unroll
        for (int r = 0; r < IQ_R; ++r) {
            const uint row = min(row0 + (uint) r, n_rows - 1u);
            __global const uchar* wr = w + (size_t) row * (size_t) wrow_bytes;
            const uint base = kbx * (uint) (IQ_FMT == 42 ? 18 : (IQ_FMT == 22 ? 82 : (IQ_FMT == 21 ? 110 : 136)));

            int g[8];
            group_decode(wr, base, m, iqs, t2, t3, kv, g);

            // block_q2_0's d is a ggml_half like every other block here - reading 4 bytes of float at
            // the block start (with the qs bytes riding along) is self-consistent but wrong.
            const float dw = f16_to_f(rd16(wr, base));
            acc[r] += apply(wr, base, m, iqs, g, xw, dw, dx);
        }
    }

#ifdef IQ_LMEM
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) lred[lane * IQ_R + r] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = IQ_WG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            #pragma unroll
            for (int r = 0; r < IQ_R; ++r)
                lred[lane * IQ_R + r] = lred[lane * IQ_R + r] + lred[(lane + off) * IQ_R + r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < IQ_R; ++r)
            if (row0 + (uint) r < n_rows) outv[row0 + (uint) r] = lred[r];
    }
#else
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) {
        const float total = sub_group_reduce_add(acc[r]);
        if (get_sub_group_local_id() == 0 && row0 + (uint) r < n_rows)
            outv[row0 + (uint) r] = total;
    }
#endif
}
)CLC";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int want_plat = -1;
    int repeat = 20;
    int rep = 8;
    bool sg = false;
    std::vector<int> want_fmts;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sg")) sg = true;
        else if (argv[i][0] == '-' && argv[i][1] != '\0') want_plat = std::atoi(argv[i]);
        else {
            const int v = std::atoi(argv[i]);
            if (pos++ == 0) repeat = v;
            else if (pos == 2) rep = v;
            else if (v == 21 || v == 22 || v == 23 || v == 42) want_fmts.push_back(v);
        }
    }
    if (want_fmts.empty()) want_fmts = {21, 22, 23, 42};
    if (repeat < 1) repeat = 1;
    if (rep < 1) rep = 1;

    cl_uint n_plat = 0;
    clGetPlatformIDs(0, nullptr, &n_plat);
    std::vector<cl_platform_id> plats(n_plat);
    clGetPlatformIDs(n_plat, plats.data(), nullptr);
    cl_platform_id plat = nullptr;
    for (cl_uint i = 0; i < n_plat; ++i) {
        size_t n = 0;
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, 0, nullptr, &n);
        std::vector<char> v(n + 1, 0);
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, n, v.data(), nullptr);
        if (want_plat < 0 ? std::strstr(v.data(), "Advanced Micro") != nullptr : (int) i == want_plat)
            plat = plats[i];
    }
    if (!plat) { std::printf("no matching platform\n"); return 1; }

    cl_uint nd = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    cl_device_id dev = devs[0];
    {
        size_t n = 0;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &n);
        std::vector<char> v(n + 1, 0);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, n, v.data(), nullptr);
        std::printf("device: %s   reduction: %s\n\n", v.data(), sg ? "sub_group_reduce" : "local-memory tree");
    }

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &rc);

    const char* names[4] = {"IQ3_S", "IQ2_S", "IQ4_XS", "Q2_0"};
    const int types[4] = {21, 22, 23, 42};

    std::printf("%-7s %4s %4s %5s %12s %12s %10s %10s %10s %10s\n",
                "format", "wg", "R", "reps", "max|d|", "tol", "GB/s", "us/matrix", "ms", "Mval/s", "copybad");

    for (int t : want_fmts) {
        // ---- load the bench file -------------------------------------------------------------------
        char path[512];
        std::snprintf(path, sizeof(path), "G:/01_Projects/Strata/opencl/probe/iqfmt_%d.bin", t);
        std::vector<char> file;
        if (!read_file(path, file)) { std::printf("cannot read %s\n", path); continue; }
        const uint32_t n_rows = rd_u32(file.data(), 0);
        const uint32_t row_vals = rd_u32(file.data(), 4);
        const uint32_t nb = rd_u32(file.data(), 8);
        const uint32_t ref_rows = rd_u32(file.data(), 12);
        const uint32_t fmt = rd_u32(file.data(), 16);
        const uint32_t bb = rd_u32(file.data(), 20);
        const size_t w_off = 24;
        const size_t w_bytes = (size_t) n_rows * nb * bb;
        const size_t x_off = w_off + w_bytes;
        const size_t x_blocks = row_vals / 32;
        const size_t x_bytes = x_blocks * 36;
        const size_t t2_off = x_off + x_bytes;                  // 1024 u64
        const size_t t3_off = t2_off + 8192;                    // 512 u32
        const size_t kv_off = t3_off + 2048;                    // 16 i8
        const size_t ref_off = kv_off + 16;
        if (fmt != (uint32_t) t) { std::printf("fmt mismatch in %s\n", path); continue; }

        std::vector<float> refs(ref_rows);
        std::memcpy(refs.data(), file.data() + ref_off, ref_rows * 4);

        // The bench file holds one matrix; replicate it so the dispatch is long enough to time.  The
        // weight traffic per row is unchanged, which is what GB/s is measured against.
        std::vector<char> wrep(w_bytes * (size_t) rep);
        for (int i = 0; i < rep; ++i) std::memcpy(wrep.data() + (size_t) i * w_bytes, file.data() + w_off, w_bytes);

        // xint is the uint view of the activation area: 9 u32 per q8_1 block, the qs starting at word 1.
        const size_t x_words = x_blocks * 9;
        std::vector<uint32_t> xint(x_words);
        std::vector<float> xf16(x_blocks * 2);
        for (size_t i = 0; i < x_blocks; ++i) {
            const char* p = file.data() + x_off + i * 36;
            xint[i * 9 + 0] = rd_u32(p, 0);
            for (int j = 0; j < 8; ++j) xint[i * 9 + 1 + (size_t) j] = rd_u32(p, 4 + (size_t) j * 4);
            const uint16_t d = (uint16_t) (rd_u32(p, 0) & 0xFFFFu);
            // the same bit-exact fp16 decode the kernel does, so dx comes out identical
            const uint32_t s = (d >> 15) & 1u, e = (d >> 10) & 0x1Fu, m = d & 0x3FFu;
            float f;
            if (e == 0) f = (m == 0) ? 0.0f : (float) m * 5.9604644775390625e-08f;
            else {
                const uint32_t bits = (s << 31) | ((e + 112u) << 23) | (m << 13);
                std::memcpy(&f, &bits, 4);
            }
            xf16[i * 2] = (s ? -f : f);
            xf16[i * 2 + 1] = 0.0f;
        }

        cl_mem bw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wrep.size(), wrep.data(), &rc);
        cl_mem bxi = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xint.size() * 4, xint.data(), &rc);
        cl_mem bxf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, xf16.size() * 4, xf16.data(), &rc);
        cl_mem bt2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8192, file.data() + t2_off, &rc);
        cl_mem bt3 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2048, file.data() + t3_off, &rc);
        cl_mem bkv = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16, file.data() + kv_off, &rc);
        const size_t total_rows = (size_t) n_rows * (size_t) rep;
        cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, total_rows * 4, nullptr, &rc);
        if (rc != CL_SUCCESS) { std::printf("buffers: %s\n", err_name(rc)); continue; }

        std::vector<float> got(total_rows);
        std::vector<float> got_first(total_rows);

        const int wgs[] = {32, 64, 128, 256};
        const int rows_per[] = {1, 4, 8};
        for (int wi = 0; wi < 4; ++wi) {
            for (int ri = 0; ri < 3; ++ri) {
                const int wg = wgs[wi], R = rows_per[ri];
                if (wg / 32 > 8) continue;
                const std::string src = source(t, wg, R, sg);
                const char* srcp = src.c_str();
                cl_program p = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &rc);
                if (clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr) != CL_SUCCESS) {
                    size_t n = 0;
                    clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
                    std::vector<char> log(n + 1, 0);
                    clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
                    std::printf("%-7s %4d %4d  build failed: %.300s\n", names[0], wg, R, log.data());
                    clReleaseProgram(p);
                    continue;
                }
                cl_kernel k = clCreateKernel(p, "iqfmt_row", &rc);
                cl_uint ai = 0;
                clSetKernelArg(k, ai++, sizeof(bw), &bw);
                clSetKernelArg(k, ai++, sizeof(bxi), &bxi);
                clSetKernelArg(k, ai++, sizeof(bxf), &bxf);
                clSetKernelArg(k, ai++, sizeof(bt2), &bt2);
                clSetKernelArg(k, ai++, sizeof(bt3), &bt3);
                clSetKernelArg(k, ai++, sizeof(bkv), &bkv);
                clSetKernelArg(k, ai++, sizeof(bo), &bo);
                const uint32_t wrow_bytes = nb * bb;
                clSetKernelArg(k, ai++, sizeof(wrow_bytes), &wrow_bytes);
                const uint32_t nrows32 = (uint32_t) total_rows;
                clSetKernelArg(k, ai++, sizeof(nrows32), &nrows32);
                if (!sg) clSetKernelArg(k, ai++, (size_t) wg * R * sizeof(float), nullptr);

                // The kernel indexes rows by work-group id, so the grid must be groups*wg work-items -
                // not total_rows.  Dispatching total_rows work-items launches total_rows/wg groups and
                // silently computes only the first 1/wg of the rows, which still looks right as long as
                // the rows checked against the reference fall in that prefix.
                const size_t groups = (total_rows + (size_t) R - 1) / (size_t) R;
                const size_t g[1] = {groups * (size_t) wg};
                const size_t l[1] = {(size_t) wg};
                if (clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr) != CL_SUCCESS ||
                    clFinish(q) != CL_SUCCESS) {
                    std::printf("%-7s %4d %4d  enqueue failed\n", names[0], wg, R);
                    clReleaseKernel(k); clReleaseProgram(p);
                    continue;
                }
                clEnqueueReadBuffer(q, bo, CL_TRUE, 0, total_rows * 4, got.data(), 0, nullptr, nullptr);
                std::memcpy(got_first.data(), got.data(), (size_t) n_rows * 4);

                // Every replicated copy must reproduce the first one.  If the dispatch were truncated the
                // tail would be stale and the timing would be measuring less work than it claims.
                uint64_t copy_mismatch = 0;
                size_t first_bad = 0;
                float copy_worst = 0.0f;
                for (size_t i = (size_t) n_rows; i < total_rows; ++i)
                    if (got[i] != got_first[i % (size_t) n_rows]) {
                        if (copy_mismatch == 0) first_bad = i;
                        ++copy_mismatch;
                        copy_worst = std::fmax(copy_worst,
                                               std::fabs(got[i] - got_first[i % (size_t) n_rows]));
                    }
                if (copy_mismatch && wg == 32 && R == 1) {
                    std::printf("    copy diff: first at row %zu (== %zu of copy 0) got %.9g copy0 %.9g "
                                "maxabs %.9g\n",
                                first_bad, first_bad % (size_t) n_rows, (double) got[first_bad],
                                (double) got_first[first_bad % (size_t) n_rows], (double) copy_worst);
                    std::printf("      bad rows:");
                    int shown = 0;
                    for (size_t i = (size_t) n_rows; i < total_rows && shown < 8; ++i)
                        if (got[i] != got_first[i % (size_t) n_rows]) {
                            std::printf(" %zu(got %.4g)", i, (double) got[i]);
                            ++shown;
                        }
                    std::printf("\n");
                }

                float worst = 0.0f, scale = 1e-30f;
                for (uint32_t i = 0; i < ref_rows; ++i) {
                    worst = std::fmax(worst, std::fabs(got[i] - refs[i]));
                    scale = std::fmax(scale, std::fabs(refs[i]));
                }
                const char* tag = names[0];
                for (int z = 0; z < 4; ++z) if (types[z] == t) tag = names[z];

                for (int i = 0; i < 2; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
                clFinish(q);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < repeat; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
                clFinish(q);
                const auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
                const double bytes = (double) total_rows * (double) nb * (double) bb;
                const double per_matrix_us = ms * 1e3 / (double) rep;

                // early rows are exact; the replicated tail is only there to be timed
                const bool exact = worst == 0.0f;
                std::printf("%-7s %4d %4d %5d %12.6f %12.3f %10.1f %10.2f %10.4f %10.1f %8llu%s\n",
                            tag, wg, R, rep, (double) worst, (double) scale, bytes / (ms * 1e-3) / 1e9,
                            per_matrix_us, ms,
                            (double) total_rows * row_vals / (ms * 1e-3) / 1e6,
                            (unsigned long long) copy_mismatch, exact ? "   EXACT" : "");
                clReleaseKernel(k);
                clReleaseProgram(p);
            }
        }

        // what the first configuration actually produced, when it does not match
        {
            float worst = 0.0f;
            uint32_t wi = 0;
            for (uint32_t i = 0; i < ref_rows; ++i)
                if (std::fabs(got_first[i] - refs[i]) > worst) { worst = (float) std::fabs(got_first[i] - refs[i]); wi = i; }
            if (worst > 0.0f)
                std::printf("    worst row %u: got %.6f  ref %.6f  (diff %.6f)\n", wi, (double) got_first[wi],
                            (double) refs[wi], (double) worst);
        }

        clReleaseMemObject(bw); clReleaseMemObject(bxi); clReleaseMemObject(bxf);
        clReleaseMemObject(bt2); clReleaseMemObject(bt3); clReleaseMemObject(bkv); clReleaseMemObject(bo);
    }

    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
