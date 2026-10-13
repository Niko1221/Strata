// opencl/probe/ocl_iq3.cpp - the real IQ3_XXS gate/up row kernel, on the MI50, via OpenCL.
//
// This is the transcription of vulkan/probe/iq3_fast.comp (itself a bit-exact transcription of
// Split<18>::load/apply in src/kernels/cuda/iq_kernels.cu) into OpenCL C.  Nothing about the arithmetic
// changes - same block decoding, same integer chain, same lane assignment, same summation order - so the
// same Python reference (vulkan/probe/iq3_ref.py, whose output is baked into iq_bench.bin) still applies.
//
// The things that had to be re-decided for OpenCL:
//
//   1. the 4x8 dot.  Vulkan's dotPacked4x8EXT does not resolve in this glslc, and OpenCL's
//      dot_acc_sat_4x8packed_ss_int is gated behind cl_khr_integer_dot_product, which gfx906 does not
//      advertise; amd_mqsad aborts the driver's BRIG printer.  Four mad24s with a sign-extended byte each
//      are what is left, and ocl_dot.cpp showed they are bit-exact and reach ~175 GB/s on a row of this
//      shape - twice the 83 GB/s the Vulkan fallback managed.
//   2. the row reduction.  Vulkan used subgroupAdd with a 32-lane subgroup for an NVIDIA wave and 64 for a
//      gfx906 wavefront, and only the 32-lane version reproduced the reference tree.  Both spellings are
//      exposed here (subgroup reduce / a local-memory tree) so the summation order can be checked rather
//      than argued about.
//   3. FP contraction.  The reference sums `acc + (dw*dx)*sumi` with each product rounded separately;
//      letting the compiler fuse the multiply-add changes the last bits.  #pragma OPENCL FP_CONTRACT OFF is
//      applied for that reason, and max_err in the output says whether it worked.
//
// Reads vulkan/probe/iq_bench.bin (real shape: 1280 gate+up rows of n_embd 2560, split d/aux/qs layout,
// 32 reference rows from the Python model).  Usage: ocl_iq3 [repeat] [-1|0|1]

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
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
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
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

// ---- the llama.cpp integer helpers, unchanged from the Vulkan kernel ------------------------------------
const char* kHelpers = R"CLC(
// bit-exact fp16 -> fp32 (exp2() is an approximation on some drivers)
inline float f16_to_f(uint h) {
    const uint s = (h >> 15u) & 1u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 31u) return as_float((s << 31u) | 0x7F800000u | (m << 13u));
    if (e == 0u) {
        if (m == 0u) return as_float(s << 31u);
        return (float) m * 5.9604644775390625e-08f;
    }
    return as_float((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// an fp16 sitting at a 2-byte-aligned byte offset inside a uint array: one load, no byte loop
inline float f16_at(__global const uint* dblob, uint byteoff) {
    return f16_to_f((dblob[byteoff >> 2] >> ((byteoff & 2u) * 8u)) & 0xFFFFu);
}

inline uint unpack_ksigns(uint v) {
    uint p = (uint) popcount(v & 0xFFu) & 1u;
    return (v & 0xFFu) ^ (p << 7u);
}

// "is each byte non-zero", giving 0xFF or 0x00 per byte.
//
// The bit trick (llama.cpp's, and the one the Vulkan kernel uses) is:
//     uint nz = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
//     return (nz >> 7) * 0xFF;
// On this driver it comes out wrong in bit 31 only - byte 3 of the mask loses its top bit, which turns a
// sign byte into a positive one.  ocl_iq3.cpp's per-sub-chunk dump shows it directly:
//     m0 (kernel) = 00ff00ff   for signs & 0x08040201 == 0x08040001
//     m0 (python) = ffff00ff
// and Python agrees with the reference row value, so the driver is at fault, not the model.
// IQ_VCNE_PLAIN therefore defaults on: the byte loop is 8 instructions instead of 5, but it is exact.
#ifndef IQ_VCNE_PLAIN
#define IQ_VCNE_PLAIN 1
#endif

inline int vcmpne4(int a, int b) {
    uint t = as_uint(a) ^ as_uint(b);
#if IQ_VCNE_PLAIN
    uint r = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (((t >> (8 * i)) & 0xFFu) != 0u) r |= 0xFFu << (8 * i);
    return as_int(r);
#else
    uint nz = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return as_int((nz >> 7u) * 0xFFu);
#endif
}

inline int vsub4(int a, int b) {
    uint ua = as_uint(a);
    uint ub = as_uint(b);
    return as_int(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

// Four 24-bit multiplies with a sign-extended byte each.  Bit-exact against __dp4a, and the fastest form
// this driver offers (see ocl_dot.cpp).
inline int dp4x8(int a, int b, int c) {
    int s = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        s = mad24((a << (24 - 8 * i)) >> 24, (b << (24 - 8 * i)) >> 24, s);
    return s;
}
)CLC";

const char* kKernel = R"CLC(
// The chunk count is fixed at build time (80 for a gate/up row), which lets the loop unroll.
#define IQ_NCHUNK 80

__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq3_row(__global const uint* dblob, __global const uint* aux, __global const uint* qsblob,
             __global const uint* cb, __global const uint* xdblob, __global const uint* xqblob,
             __global float* outv, const uint nb, const uint n_rows
#ifdef IQ_LMEM
             , __local float* lred
#endif
             ) {
    const uint lane = get_local_id(0);
    const uint row0 = get_group_id(0) * IQ_R;
    const uint n_chunks = IQ_NCHUNK;

    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

    for (uint k = lane; k < n_chunks; k += IQ_WG) {
        const uint kbx = k >> 3;                 // 256-value block
        const uint c = k & 7u;                   // 32-value chunk inside it, 8 bytes of qs
        const uint xb = (kbx * 8u + c) * 8u;     // q8_1 block: 8 uints of int8

        // Read the activation once and feed all R rows; at ~3 bits per weight the activation traffic is
        // several times the weight traffic, so this is where the win is.
        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = xqblob[xb + (uint) j];
        const uint xchunk = kbx * 8u + c;
        const float dx = f16_to_f((xdblob[xchunk >> 1u] >> ((xchunk & 1u) * 16u)) & 0xFFFFu);

        #pragma unroll
        for (int r = 0; r < IQ_R; ++r) {
            const uint row = min(row0 + (uint) r, n_rows - 1u);
            const uint qs_row = row * n_chunks;   // in units of 8 bytes
            const uint a32 = aux[row * n_chunks + k];

            const uint qlo = qsblob[(qs_row + k) * 2u];
            const uint qhi = qsblob[(qs_row + k) * 2u + 1u];

            int g[8];
            #pragma unroll
            for (int li = 0; li < 4; ++li) {
                const int l0 = li * 2;
                const uint w = (l0 < 4) ? qlo : qhi;
                const uint sh0 = (l0 < 4) ? (uint) (8 * l0) : (uint) (8 * l0 - 32);
                const uint gx = cb[(w >> sh0) & 0xFFu];
                const uint gy = cb[(w >> (sh0 + 8u)) & 0xFFu];
                const uint signs = unpack_ksigns(a32 >> (uint) (7 * l0 / 2)) * 0x01010101u;
                const int m0 = vcmpne4((int) (signs & 0x08040201u), 0);
                const int m1 = vcmpne4((int) (signs & 0x80402010u), 0);
                g[l0] = vsub4((int) gx ^ m0, m0);
                g[l0 + 1] = vsub4((int) gy ^ m1, m1);
            }

            int si = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);

            const int sumi = ((int) (a32 >> 28u) * si + si / 2) / 2;
            acc[r] += f16_at(dblob, row * nb * 2u + kbx * 2u) * dx * (float) sumi;
        }
    }

#ifdef IQ_LMEM
    // The reference's own tree, spelled out: IQ_WG/2, ... 4, 2, 1.  Every row is reduced at once so the
    // barriers are paid once per work-group rather than once per row.
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
    // sub_group_reduce_add is a collective, so every lane must reach it; the lane-0 store comes after.
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) {
        const float total = sub_group_reduce_add(acc[r]);
        if (get_sub_group_local_id() == 0 && row0 + (uint) r < n_rows)
            outv[row0 + (uint) r] = total;
    }
#endif
}

// Reported once so the subgroup-spelling results can be read with the right expectation.
__kernel void sgsize(__global uint* o) {
    o[0] = (uint) get_sub_group_size();
}

// Per-chunk dissection of one row: one work-item per chunk, writing the integer product, the block scale
// and the activation scale.  When the whole-row answer is wrong, this says which of the three is wrong
// instead of leaving it to guesswork.
__kernel void iq3_dbg(__global const uint* dblob, __global const uint* aux, __global const uint* qsblob,
                      __global const uint* cb, __global const uint* xdblob, __global const uint* xqblob,
                      __global int* out_si, __global uint* out_raw, __global float* out_sc,
                      __global int* out_g, __global uint* out_w, __global uint* out_li,
                      const uint nb, const uint row) {
    const uint k = get_global_id(0);
    if (k >= IQ_NCHUNK) return;
    const uint kbx = k >> 3;
    const uint c = k & 7u;
    const uint xb = (kbx * 8u + c) * 8u;

    uint xw[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) xw[j] = xqblob[xb + (uint) j];
    const float dx = f16_to_f((xdblob[k >> 1u] >> ((k & 1u) * 16u)) & 0xFFFFu);

    const uint qs_row = row * IQ_NCHUNK;
    const uint a32 = aux[row * IQ_NCHUNK + k];
    const uint qlo = qsblob[(qs_row + k) * 2u];
    const uint qhi = qsblob[(qs_row + k) * 2u + 1u];

    int g[8];
    #pragma unroll
    for (int li = 0; li < 4; ++li) {
        const int l0 = li * 2;
        const uint w = (l0 < 4) ? qlo : qhi;
        const uint sh0 = (l0 < 4) ? (uint) (8 * l0) : (uint) (8 * l0 - 32);
        const uint gx = cb[(w >> sh0) & 0xFFu];
        const uint gy = cb[(w >> (sh0 + 8u)) & 0xFFu];
        const uint signs = unpack_ksigns(a32 >> (uint) (7 * l0 / 2)) * 0x01010101u;
        const int m0 = vcmpne4((int) (signs & 0x08040201u), 0);
        const int m1 = vcmpne4((int) (signs & 0x80402010u), 0);
        // the four inputs and the two masks, so a mismatch can be attributed byte by byte
        out_li[(k * 4u + (uint) li) * 8u + 0u] = signs;
        out_li[(k * 4u + (uint) li) * 8u + 1u] = gx;
        out_li[(k * 4u + (uint) li) * 8u + 2u] = gy;
        out_li[(k * 4u + (uint) li) * 8u + 3u] = (uint) m0;
        out_li[(k * 4u + (uint) li) * 8u + 4u] = (uint) m1;
        out_li[(k * 4u + (uint) li) * 8u + 5u] = (uint) sh0;
        out_li[(k * 4u + (uint) li) * 8u + 6u] = (w >> sh0) & 0xFFu;
        out_li[(k * 4u + (uint) li) * 8u + 7u] = (w >> (sh0 + 8u)) & 0xFFu;
        g[l0] = vsub4((int) gx ^ m0, m0);
        g[l0 + 1] = vsub4((int) gy ^ m1, m1);
    }

    int si = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        out_g[k * 8u + (uint) j] = g[j];
        out_w[k * 8u + (uint) j] = xw[j];
        si = dp4x8(g[j], (int) xw[j], si);
    }

    const int sumi = ((int) (a32 >> 28u) * si + si / 2) / 2;
    out_si[k] = si;
    out_raw[k * 4u + 0u] = qlo;
    out_raw[k * 4u + 1u] = qhi;
    out_raw[k * 4u + 2u] = a32;
    out_raw[k * 4u + 3u] = (uint) sumi;
    // the Python model's own decomposition, so the scales can be compared one at a time
    out_sc[k * 3u + 0u] = f16_at(dblob, row * nb * 2u + kbx * 2u);
    out_sc[k * 3u + 1u] = dx;
    out_sc[k * 3u + 2u] = f16_at(dblob, row * nb * 2u + kbx * 2u) * dx * (float) sumi;
}
)CLC";

/// Assemble the program: macros first, then the helpers (which use kHigh, so it has to be defined before
/// them), then the kernel.
std::string iq3_source(int wg, int rows_per_lane, bool subgroup_reduce) {
    std::string s;
    if (!subgroup_reduce) s += "#define IQ_LMEM 1\n";
    s += "#define IQ_WG " + std::to_string(wg) + "\n";
    s += "#define IQ_R " + std::to_string(rows_per_lane) + "\n";
    s += "#define kHigh 0x80808080u\n";
    s += "#pragma OPENCL FP_CONTRACT OFF\n";
    if (subgroup_reduce) s += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
    s += kHelpers;
    s += kKernel;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    int want_plat = -1;
    int repeat = 20;
    int mul = 1;                 // weight buffers are replicated `mul` times so the dispatch is big
    int pos = 0;                 // enough to time: at 1x the launch overhead is most of the measurement
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') want_plat = std::atoi(argv[i]);
        else if (pos++ == 0) repeat = std::atoi(argv[i]);
        else mul = std::atoi(argv[i]);
    }

    // ---- data -----------------------------------------------------------------------------------------
    std::vector<char> raw;
    const char* paths[] = {"../../vulkan/probe/iq_bench.bin", "../vulkan/probe/iq_bench.bin",
                           "G:/01_Projects/Strata/vulkan/probe/iq_bench.bin"};
    bool loaded = false;
    for (const char* p : paths)
        if (read_file(p, raw) && raw.size() > 64) { std::printf("data:  %s\n", p); loaded = true; break; }
    if (!loaded) { std::printf("cannot find iq_bench.bin\n"); return 1; }

    const uint32_t* hdr = (const uint32_t*) raw.data();
    const unsigned n_rows = hdr[0], n_embd = hdr[1], nb = hdr[2], ref_rows = hdr[3];
    const size_t d_bytes = (size_t) n_rows * nb * 2;
    const size_t aux_bytes = (size_t) n_rows * nb * 8 * 4;
    const size_t qs_bytes = (size_t) n_rows * nb * 64;
    const unsigned n_chunks = n_embd / 32;
    const size_t xd_bytes = (size_t) (((n_chunks + 1) / 2) * 4);
    const size_t xq_bytes = (size_t) n_chunks * 8 * 4;
    size_t off = 16;
    const char* d_src = raw.data() + off; off += d_bytes;
    const char* aux_src = raw.data() + off; off += aux_bytes;
    const char* qs_src = raw.data() + off; off += qs_bytes;
    const char* xd_src = raw.data() + off; off += xd_bytes;
    const char* xq_src = raw.data() + off; off += xq_bytes;
    const char* cb_src = raw.data() + off; off += 256 * 4;
    if (off + (size_t) ref_rows * 4 > raw.size()) { std::printf("iq_bench.bin is truncated\n"); return 1; }
    const float* refs = (const float*) (raw.data() + off);
    const unsigned rows_total = n_rows * (unsigned) mul;
    std::printf("shape: %u rows, n_embd %u, nb %u, %u refs, weight %.3f MB\n",
                n_rows, n_embd, nb, ref_rows, (double) (d_bytes + aux_bytes + qs_bytes) / 1e6);

    // ---- device ---------------------------------------------------------------------------------------
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
        std::printf("device: %s\n", v.data());
    }

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &rc);

#define MK_BUF(name, bytes, ptr, flags)                                                                     \
    cl_mem name = clCreateBuffer(ctx, (flags) | CL_MEM_COPY_HOST_PTR, (size_t) (bytes), (void*) (ptr), &rc); \
    if (rc != CL_SUCCESS) { std::printf("buffer " #name ": %s\n", err_name(rc)); return 1; }
    // the three row-dependent arrays are replicated; the activation and the codebook are shared
    auto rep = [mul](const char* src, size_t bytes) {
        std::vector<char> v(bytes * (size_t) mul);
        for (int i = 0; i < mul; ++i) std::memcpy(v.data() + (size_t) i * bytes, src, bytes);
        return v;
    };
    const std::vector<char> d_rep = rep(d_src, d_bytes);
    const std::vector<char> aux_rep = rep(aux_src, aux_bytes);
    const std::vector<char> qs_rep = rep(qs_src, qs_bytes);
    MK_BUF(bd, d_bytes * (size_t) mul, d_rep.data(), CL_MEM_READ_ONLY)
    MK_BUF(baux, aux_bytes * (size_t) mul, aux_rep.data(), CL_MEM_READ_ONLY)
    MK_BUF(bqs, qs_bytes * (size_t) mul, qs_rep.data(), CL_MEM_READ_ONLY)
    MK_BUF(bcb, 256 * 4, cb_src, CL_MEM_READ_ONLY)
    MK_BUF(bxd, xd_bytes, xd_src, CL_MEM_READ_ONLY)
    MK_BUF(bxq, xq_bytes, xq_src, CL_MEM_READ_ONLY)
#undef MK_BUF
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) rows_total * 4, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("buffer out: %s\n", err_name(rc)); return 1; }

    // ---- what is a subgroup here? ---------------------------------------------------------------------
    {
        const std::string s0 = iq3_source(64, 1, true);
        const char* p0 = s0.c_str();
        cl_program pr = clCreateProgramWithSource(ctx, 1, &p0, nullptr, &rc);
        if (clBuildProgram(pr, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr) == CL_SUCCESS) {
            cl_kernel ks = clCreateKernel(pr, "sgsize", &rc);
            if (rc == CL_SUCCESS) {
                cl_mem ob = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 4, nullptr, &rc);
                clSetKernelArg(ks, 0, sizeof(ob), &ob);
                const size_t g[1] = {64}, l[1] = {64};
                if (clEnqueueNDRangeKernel(q, ks, 1, nullptr, g, l, 0, nullptr, nullptr) == CL_SUCCESS) {
                    clFinish(q);
                    cl_uint got = 0;
                    clEnqueueReadBuffer(q, ob, CL_TRUE, 0, 4, &got, 0, nullptr, nullptr);
                    std::printf("subgroup size reported by the device: %u\n", got);
                }
                clReleaseMemObject(ob);
                clReleaseKernel(ks);
            }
        }
        clReleaseProgram(pr);
    }

    // ---- dissect one row, chunk by chunk, against the Python model -------------------------------------
    {
        const std::string src0 = iq3_source(32, 1, true);
        const char* p0 = src0.c_str();
        cl_program pr = clCreateProgramWithSource(ctx, 1, &p0, nullptr, &rc);
        if (clBuildProgram(pr, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr) == CL_SUCCESS) {
            cl_kernel kd = clCreateKernel(pr, "iq3_dbg", &rc);
            if (rc == CL_SUCCESS) {
                cl_mem bsi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 4, nullptr, &rc);
                cl_mem braw = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 16, nullptr, &rc);
                cl_mem bsc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 12, nullptr, &rc);
                cl_mem bg = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 32, nullptr, &rc);
                cl_mem bw = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 32, nullptr, &rc);
                cl_mem bli = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_chunks * 4 * 32, nullptr, &rc);
                // the extra binding shifts every argument after it by one
                const unsigned row0 = 0;
                clSetKernelArg(kd, 0, sizeof(bd), &bd);
                clSetKernelArg(kd, 1, sizeof(baux), &baux);
                clSetKernelArg(kd, 2, sizeof(bqs), &bqs);
                clSetKernelArg(kd, 3, sizeof(bcb), &bcb);
                clSetKernelArg(kd, 4, sizeof(bxd), &bxd);
                clSetKernelArg(kd, 5, sizeof(bxq), &bxq);
                clSetKernelArg(kd, 6, sizeof(bsi), &bsi);
                clSetKernelArg(kd, 7, sizeof(braw), &braw);
                clSetKernelArg(kd, 8, sizeof(bsc), &bsc);
                clSetKernelArg(kd, 9, sizeof(bg), &bg);
                clSetKernelArg(kd, 10, sizeof(bw), &bw);
                clSetKernelArg(kd, 11, sizeof(bli), &bli);
                clSetKernelArg(kd, 12, sizeof(nb), &nb);
                clSetKernelArg(kd, 13, sizeof(row0), &row0);
                const size_t g[1] = {(size_t) n_chunks};
                if (clEnqueueNDRangeKernel(q, kd, 1, nullptr, g, nullptr, 0, nullptr, nullptr) == CL_SUCCESS) {
                    clFinish(q);
                    std::vector<cl_int> si(n_chunks);
                    std::vector<cl_uint> raw((size_t) n_chunks * 4);
                    std::vector<float> sc((size_t) n_chunks * 3);
                    std::vector<cl_int> gv((size_t) n_chunks * 8);
                    std::vector<cl_uint> wv((size_t) n_chunks * 8);
                    clEnqueueReadBuffer(q, bsi, CL_TRUE, 0, (size_t) n_chunks * 4, si.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(q, braw, CL_TRUE, 0, (size_t) n_chunks * 16, raw.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(q, bsc, CL_TRUE, 0, (size_t) n_chunks * 12, sc.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(q, bg, CL_TRUE, 0, (size_t) n_chunks * 32, gv.data(), 0, nullptr, nullptr);
                    clEnqueueReadBuffer(q, bw, CL_TRUE, 0, (size_t) n_chunks * 32, wv.data(), 0, nullptr, nullptr);
                    std::vector<cl_uint> li((size_t) n_chunks * 4 * 8);
                    clEnqueueReadBuffer(q, bli, CL_TRUE, 0, (size_t) n_chunks * 4 * 32, li.data(), 0, nullptr, nullptr);
                    for (unsigned k = 0; k < 1; ++k)
                        for (unsigned s = 0; s < 4; ++s) {
                            const cl_uint* v = &li[(k * 4 + s) * 8];
                            std::printf("        li=%u signs=%08x gx=%08x gy=%08x m0=%08x m1=%08x sh0=%u idx=%02x,%02x -> g[%u]=%d g[%u]=%d\n",
                                        s, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                        s * 2, gv[k * 8 + s * 2], s * 2 + 1, gv[k * 8 + s * 2 + 1]);
                        }
                    std::printf("\nrow 0 dissection (first 4 of %u chunks):\n", n_chunks);
                    for (unsigned k = 0; k < 4 && k < n_chunks; ++k) {
                        std::printf("  k=%u  qlo=%08x qhi=%08x a32=%08x  s=%d  sumi=%u  dw=%.8g dx=%.8g term=%.8g\n",
                                    k, raw[k * 4], raw[k * 4 + 1], raw[k * 4 + 2], si[k], raw[k * 4 + 3],
                                    (double) sc[k * 3], (double) sc[k * 3 + 1], (double) sc[k * 3 + 2]);
                        std::printf("        g =");
                        for (int j = 0; j < 8; ++j) std::printf(" %11d", gv[k * 8 + j]);
                        std::printf("\n        xw=");
                        for (int j = 0; j < 8; ++j) std::printf(" %11u", wv[k * 8 + j]);
                        std::printf("\n");
                    }
                }
                clReleaseMemObject(bsi);
                clReleaseMemObject(braw);
                clReleaseMemObject(bsc);
                clReleaseMemObject(bg);
                clReleaseMemObject(bw);
                clReleaseMemObject(bli);
                clReleaseKernel(kd);
            }
        }
        clReleaseProgram(pr);
    }

    // ---- sweep configurations -------------------------------------------------------------------------
    struct Cfg { int wg; int r; bool subgroup; };
    const Cfg cfgs[] = {
        {32, 1, true}, {32, 4, true}, {32, 8, true},
        {64, 1, true}, {64, 4, true}, {64, 8, true},
        {32, 4, false}, {32, 8, false},
    };
    std::vector<float> hout(rows_total, 0.0f);
    const double w_bytes = (double) (d_bytes + aux_bytes + qs_bytes) * (double) mul;

    std::printf("\ndispatch: %u rows (%d copies of %u), %.1f MB of weights\n", rows_total, mul, n_rows, w_bytes / 1e6);
    std::printf("%-20s %11s %10s %10s %9s %9s %10s\n", "config", "max_err", "bad/refs", "ms",
                "GB/s", "Mrows/s", "us/expert");
    for (const Cfg& c : cfgs) {
        const std::string src = iq3_source(c.wg, c.r, c.subgroup);
        const char* srcp = src.c_str();
        cl_int crc = CL_SUCCESS;
        cl_program p = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &crc);
        if (crc != CL_SUCCESS) { std::printf("cfg wg=%d r=%d: source %s\n", c.wg, c.r, err_name(crc)); continue; }
        crc = clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
        if (crc != CL_SUCCESS) {
            std::printf("wg=%-3d r=%-2d %-7s BUILD FAILED %s\n", c.wg, c.r, c.subgroup ? "subgrp" : "lmem",
                        err_name(crc));
            size_t n = 0;
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n + 1, 0);
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            std::printf("      %s\n", log.data());
            clReleaseProgram(p);
            continue;
        }
        cl_kernel k = clCreateKernel(p, "iq3_row", &crc);
        if (crc != CL_SUCCESS) {
            std::printf("wg=%d r=%d: kernel %s\n", c.wg, c.r, err_name(crc));
            clReleaseProgram(p);
            continue;
        }

        clSetKernelArg(k, 0, sizeof(bd), &bd);
        clSetKernelArg(k, 1, sizeof(baux), &baux);
        clSetKernelArg(k, 2, sizeof(bqs), &bqs);
        clSetKernelArg(k, 3, sizeof(bcb), &bcb);
        clSetKernelArg(k, 4, sizeof(bxd), &bxd);
        clSetKernelArg(k, 5, sizeof(bxq), &bxq);
        clSetKernelArg(k, 6, sizeof(bout), &bout);
        clSetKernelArg(k, 7, sizeof(nb), &nb);
        clSetKernelArg(k, 8, sizeof(rows_total), &rows_total);
        if (!c.subgroup) {
            const size_t lmem = (size_t) c.wg * (size_t) c.r * sizeof(float);
            clSetKernelArg(k, 9, lmem, nullptr);
        }

        const size_t groups = (rows_total + (size_t) c.r - 1) / (size_t) c.r;
        const size_t g[1] = {groups * (size_t) c.wg};
        const size_t l[1] = {(size_t) c.wg};
        crc = clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        if (crc != CL_SUCCESS) {
            std::printf("wg=%d r=%d: enqueue %s\n", c.wg, c.r, err_name(crc));
            clReleaseKernel(k);
            clReleaseProgram(p);
            continue;
        }
        clFinish(q);

        clEnqueueReadBuffer(q, bout, CL_TRUE, 0, (size_t) rows_total * 4, hout.data(), 0, nullptr, nullptr);
        double max_err = 0;
        int bad = 0;
        for (unsigned r = 0; r < ref_rows; ++r) {
            const double e = std::fabs((double) hout[r] - (double) refs[r]);
            if (e > max_err) max_err = e;
            if (e > 1e-4 * (std::fabs((double) refs[r]) + 1e-3)) ++bad;
        }

        for (int i = 0; i < 3; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        clFinish(q);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        clFinish(q);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

        char label[48];
        std::snprintf(label, sizeof(label), "wg=%d r=%d %s", c.wg, c.r, c.subgroup ? "subgroup" : "lmem");
        // one expert's gate+up is 1280 rows, so us/expert is the number that matters for the tier
        std::printf("%-20s %11.6f %7d/%-3u %10.4f %9.1f %9.1f %10.2f\n", label, max_err, bad, ref_rows, ms,
                    w_bytes / (ms * 1e-3) / 1e9, (double) rows_total / (ms * 1e-3) / 1e6,
                    ms * 1000.0 / (double) rows_total * 1280.0);
        clReleaseKernel(k);
        clReleaseProgram(p);
    }
    return 0;
}
