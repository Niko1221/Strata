// opencl/probe/ocl_iq4nl.cpp - the IQ4_NL down-projection kernel, on the MI50, via OpenCL.
//
// The second of the expert blob's three matrices.  Down maps n_ff back to n_embd: 2560 rows of 640
// values, i.e. 360 bytes a row at IQ4_NL's 18 bytes per 32 values, and 0.922 MB for the matrix.  The
// real pack uses IQ4_NL for 39 of the 48 layers (native_experts.txt), so this is the matrix that
// carries most of the down-projection traffic.
//
// Transcribed from vulkan/probe/iq4nl.comp, which was itself verified bit-exactly against
// vulkan/probe/iq4nl_ref.py.  Two things differ from the IQ3_XXS kernel beyond the format:
//
//   * the codebook is non-linear (kvalues_iq4nl = -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)
//     and arrives packed four int8 per uint, so a nibble is looked up and sign-extended rather than
//     computed.  There are only 2 dot calls per 18-byte block instead of 8, so the lookup cost is
//     proportionally larger here.
//   * `q0`'s low nibbles feed activation word iqs+0 and its high nibbles iqs+4; `q1` feeds iqs+1 and
//     iqs+5.  That is the same four-term chain, just with the pairs interleaved.
//
// The self-check in iq4nl_ref.py compares the integer chain against a naive dequantisation, and it
// passes at 3e-6, so the reference itself is trustworthy.
//
// Usage: ocl_iq4nl [repeat] [copies] [-1|0|1]

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

const char* kHelpers = R"CLC(
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

// IQ4_NL's block scale is an fp16 at the start of each 18-byte block, so d is its own array here
inline float f16_at(__global const uint* dblob, uint byteoff) {
    return f16_to_f((dblob[byteoff >> 2] >> ((byteoff & 2u) * 8u)) & 0xFFFFu);
}

// nibble (0..15) -> its codebook int8, sign-extended; the table arrives as four int8 per uint
inline int tbl(__global const uint* tab, uint nib) {
    const uint b = (tab[nib >> 2u] >> ((nib & 3u) * 8u)) & 0xFFu;
    return as_int(b << 24u) >> 24;
}

// four nibbles of one dword, low or high, packed back into one dp4a operand
inline int expand(__global const uint* tab, uint q4, int high) {
    const uint sh = high ? 4u : 0u;
    int r = 0;
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i)
        r |= (tbl(tab, (q4 >> (8u * i + sh)) & 0xFu) & 0xFF) << (int) (8u * i);
    return r;
}

// Four 24-bit multiplies with a sign-extended byte each; bit-exact against __dp4a (see ocl_dot.cpp)
inline int dp4x8(int a, int b, int c) {
    int s = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        s = mad24((a << (24 - 8 * i)) >> 24, (b << (24 - 8 * i)) >> 24, s);
    return s;
}
)CLC";

const char* kKernel = R"CLC(
__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq4nl_row(__global const uint* dblob, __global const uint* qsblob, __global const uint* tab,
               __global const uint* xdblob, __global const uint* xqblob,
               __global float* outv, const uint nb, const uint n_rows
#ifdef IQ_LMEM
               , __local float* lred
#endif
               ) {
    const uint lane = get_local_id(0);
    const uint row0 = get_group_id(0) * IQ_R;
    const uint n_calls = nb * 2u;          // 40 for a 640-wide row

    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

    for (uint k = lane; k < n_calls; k += IQ_WG) {
        const uint kbx = k >> 1;
        const uint iqs = (k & 1u) * 2u;

        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = xqblob[kbx * 8u + (uint) j];
        const float dx = f16_to_f((xdblob[kbx >> 1u] >> ((kbx & 1u) * 16u)) & 0xFFFFu);

        #pragma unroll
        for (int r = 0; r < IQ_R; ++r) {
            const uint row = min(row0 + (uint) r, n_rows - 1u);
            // four dwords per block, so dword j of block kbx sits at (row*nb + kbx)*4 + j
            const uint q0 = qsblob[(row * nb + kbx) * 4u + iqs + 0u];
            const uint q1 = qsblob[(row * nb + kbx) * 4u + iqs + 1u];

            int s = 0;
            s = dp4x8(expand(tab, q0, 0), (int) xw[iqs + 0u], s);
            s = dp4x8(expand(tab, q0, 1), (int) xw[iqs + 4u], s);
            s = dp4x8(expand(tab, q1, 0), (int) xw[iqs + 1u], s);
            s = dp4x8(expand(tab, q1, 1), (int) xw[iqs + 5u], s);

            acc[r] += f16_at(dblob, row * nb * 2u + kbx * 2u) * dx * (float) s;
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

std::string iq4nl_source(int wg, int rows_per_lane, bool subgroup_reduce) {
    std::string s;
    if (!subgroup_reduce) s += "#define IQ_LMEM 1\n";
    s += "#define IQ_WG " + std::to_string(wg) + "\n";
    s += "#define IQ_R " + std::to_string(rows_per_lane) + "\n";
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
    int mul = 1;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') want_plat = std::atoi(argv[i]);
        else if (pos++ == 0) repeat = std::atoi(argv[i]);
        else mul = std::atoi(argv[i]);
    }

    std::vector<char> raw;
    const char* paths[] = {"../../vulkan/probe/iq4nl.bin", "../vulkan/probe/iq4nl.bin",
                           "G:/01_Projects/Strata/vulkan/probe/iq4nl.bin"};
    bool loaded = false;
    for (const char* p : paths)
        if (read_file(p, raw) && raw.size() > 64) { std::printf("data:  %s\n", p); loaded = true; break; }
    if (!loaded) { std::printf("cannot find iq4nl.bin\n"); return 1; }

    const uint32_t* hdr = (const uint32_t*) raw.data();
    const unsigned n_rows = hdr[0], n_embd = hdr[1], nb = hdr[2], ref_rows = hdr[3], tab_len = hdr[4];
    if (tab_len != 16) { std::printf("unexpected codebook length %u\n", tab_len); return 1; }
    const unsigned n_calls = nb * 2;
    const size_t d_bytes = (size_t) n_rows * nb * 2;
    const size_t qs_bytes = (size_t) n_rows * nb * 16;
    const size_t xd_bytes = (size_t) nb * 2;
    const size_t xq_bytes = (size_t) nb * 8 * 4;
    size_t off = 20;
    const char* d_src = raw.data() + off; off += d_bytes;
    const char* qs_src = raw.data() + off; off += qs_bytes;
    const char* xd_src = raw.data() + off; off += xd_bytes;
    const char* xq_src = raw.data() + off; off += xq_bytes;
    const char* tab_src = raw.data() + off; off += 16;
    if (off + (size_t) ref_rows * 4 > raw.size()) { std::printf("iq4nl.bin is truncated\n"); return 1; }
    const float* refs = (const float*) (raw.data() + off);
    const unsigned rows_total = n_rows * (unsigned) mul;
    std::printf("shape: %u down rows, n_ff %u, nb %u, %u calls/row, %u refs, weight %.3f MB\n",
                n_rows, n_embd, nb, n_calls, ref_rows, (double) (d_bytes + qs_bytes) / 1e6);

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

    auto rep = [mul](const char* src, size_t bytes) {
        std::vector<char> v(bytes * (size_t) mul);
        for (int i = 0; i < mul; ++i) std::memcpy(v.data() + (size_t) i * bytes, src, bytes);
        return v;
    };
    const std::vector<char> d_rep = rep(d_src, d_bytes);
    const std::vector<char> qs_rep = rep(qs_src, qs_bytes);

#define MK_BUF(name, bytes, ptr)                                                                       \
    cl_mem name = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t) (bytes),      \
                                 (void*) (ptr), &rc);                                                  \
    if (rc != CL_SUCCESS) { std::printf("buffer " #name ": %s\n", err_name(rc)); return 1; }
    MK_BUF(bd, d_bytes * (size_t) mul, d_rep.data())
    MK_BUF(bqs, qs_bytes * (size_t) mul, qs_rep.data())
    MK_BUF(btab, 16, tab_src)
    MK_BUF(bxd, xd_bytes, xd_src)
    MK_BUF(bxq, xq_bytes, xq_src)
#undef MK_BUF
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) rows_total * 4, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("buffer out: %s\n", err_name(rc)); return 1; }

    struct Cfg { int wg; int r; bool subgroup; };
    const Cfg cfgs[] = {
        {32, 1, true}, {32, 4, true}, {32, 8, true},
        {64, 1, true}, {64, 4, true}, {64, 8, true},
        {32, 4, false}, {32, 8, false}, {64, 4, false},
    };
    std::vector<float> hout(rows_total, 0.0f);
    const double w_bytes = (double) (d_bytes + qs_bytes) * (double) mul;

    std::printf("\ndispatch: %u rows (%d copies of %u), %.1f MB of weights\n", rows_total, mul, n_rows,
                w_bytes / 1e6);
    std::printf("%-20s %11s %10s %10s %9s %9s %10s\n", "config", "max_err", "bad/refs", "ms",
                "GB/s", "Mrows/s", "us/down");
    for (const Cfg& c : cfgs) {
        const std::string src = iq4nl_source(c.wg, c.r, c.subgroup);
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
        cl_kernel k = clCreateKernel(p, "iq4nl_row", &crc);
        if (crc != CL_SUCCESS) {
            std::printf("wg=%d r=%d: kernel %s\n", c.wg, c.r, err_name(crc));
            clReleaseProgram(p);
            continue;
        }
        clSetKernelArg(k, 0, sizeof(bd), &bd);
        clSetKernelArg(k, 1, sizeof(bqs), &bqs);
        clSetKernelArg(k, 2, sizeof(btab), &btab);
        clSetKernelArg(k, 3, sizeof(bxd), &bxd);
        clSetKernelArg(k, 4, sizeof(bxq), &bxq);
        clSetKernelArg(k, 5, sizeof(bout), &bout);
        clSetKernelArg(k, 6, sizeof(nb), &nb);
        clSetKernelArg(k, 7, sizeof(rows_total), &rows_total);
        if (!c.subgroup) {
            const size_t lmem = (size_t) c.wg * (size_t) c.r * sizeof(float);
            clSetKernelArg(k, 8, lmem, nullptr);
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
        // one down matrix is 2560 rows, so us/down is the per-expert cost of this matrix
        std::printf("%-20s %11.6f %7d/%-3u %10.4f %9.1f %9.1f %10.2f\n", label, max_err, bad, ref_rows, ms,
                    w_bytes / (ms * 1e-3) / 1e9, (double) rows_total / (ms * 1e-3) / 1e6,
                    ms * 1000.0 / (double) rows_total * (double) n_rows);
        clReleaseKernel(k);
        clReleaseProgram(p);
    }
    return 0;
}
