// opencl/probe/ocl_dot.cpp - which 4x8 integer dot can the MI50 (gfx906) actually use, and what does it cost?
//
// Every i-quant expert kernel in src/kernels/cuda/iq_kernels.cu is built around __dp4a: four signed byte
// products summed into an int32.  Getting an equivalent on gfx906 is the thing the whole OpenCL plan rests
// on, so this measures rather than assumes.
//
// What the driver's own binaries say (strings scanned out of DriverStore\...\u0412788...\B413136\):
//   dot_acc_sat_4x8packed_ss_int  is in amd_comgr.dll, but the compiler only declares it when the device
//                                 advertises cl_khr_integer_dot_product.  gfx906 on this driver does not,
//                                 so the builtin is not visible to the front end.
//   amd_mqsad                     is in amdhsail64.dll / amd_comgr.dll, i.e. available - but it lowers to
//                                 V_MQSAD_PK_U16_U8, a quad-byte *SAD*, not a dot product.
//   inline asm                    is rejected outright ("undeclared identifier 'asm'"), so the instruction
//                                 cannot be forced by hand either.
//
// Which leaves four mad24s with a sign-extended byte each, or a plain signed multiply loop.  The Vulkan
// probe already reached 83 GB/s on the MI50 using its scalar fallback dot, so the missing dp4a is not
// expected to be fatal - the timing at the bottom is what decides it.
//
// Each candidate is compiled as its own program so a rejection cannot hide the others, then run against a
// host reference, then timed on the real row shape (2560 signed bytes per row).

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
        default: return "CL_ERR";
    }
}

int s8(unsigned v) { return v >= 128u ? (int) v - 256 : (int) v; }

int dot4_ref(unsigned a, unsigned b) {
    int s = 0;
    for (int k = 0; k < 4; ++k)
        s += s8((a >> (8 * k)) & 0xFFu) * s8((b >> (8 * k)) & 0xFFu);
    return s;
}

/// Compile one source, print the verdict, hand back the program.  A rejected candidate is a result, not an
/// error: the caller records it and carries on.
cl_program build_one(cl_context ctx, cl_device_id dev, const char* label, const char* src, bool& ok) {
    cl_int rc = CL_SUCCESS;
    cl_program p = clCreateProgramWithSource(ctx, 1, &src, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("  %-32s CREATE FAILED %s\n", label, err_name(rc)); ok = false; return nullptr; }
    rc = clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
    if (rc != CL_SUCCESS) {
        std::printf("  %-32s NOT BUILDABLE (%s)\n", label, err_name(rc));
        size_t n = 0;
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::vector<char> log(n + 1, 0);
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        std::string s(log.data());
        const size_t nl = s.find('\n');
        std::printf("      %s\n", s.substr(0, nl == std::string::npos ? s.size() : nl).c_str());
        clReleaseProgram(p);
        ok = false;
        return nullptr;
    }
    std::printf("  %-32s builds\n", label);
    ok = true;
    return p;
}

// ---- candidate dot forms -------------------------------------------------------------------------------

const char* kDotOcl20 = R"CLC(
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
__kernel void d(__global const uint* a, __global const uint* b, __global int* out) {
    out[get_global_id(0)] = dot_acc_sat_4x8packed_ss_int(a[get_global_id(0)], b[get_global_id(0)], 0);
}
)CLC";

const char* kDotMqsad = R"CLC(
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
__kernel void d(__global const uint* a, __global const uint* b, __global int* out) {
    out[get_global_id(0)] = (int) amd_mqsad(a[get_global_id(0)], b[get_global_id(0)], 0);
}
)CLC";

const char* kDotMad24 = R"CLC(
__kernel void d(__global const uint* a, __global const uint* b, __global int* out) {
    const int i = get_global_id(0);
    const int x = (int) a[i], y = (int) b[i];
    int s = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        s = mad24((x << (24 - 8 * k)) >> 24, (y << (24 - 8 * k)) >> 24, s);
    out[i] = s;
}
)CLC";

const char* kDotPlain = R"CLC(
__kernel void d(__global const uint* a, __global const uint* b, __global int* out) {
    const int i = get_global_id(0);
    const int x = (int) a[i], y = (int) b[i];
    int s = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        s += ((x << (24 - 8 * k)) >> 24) * ((y << (24 - 8 * k)) >> 24);
    out[i] = s;
}
)CLC";

// ---- the row kernel ------------------------------------------------------------------------------------

/// A work-group of WG lanes owns ROWS_PER_GROUP whole rows.  Within a row each lane takes every WG-th
/// 4-byte group, so the per is spread over the lanes the way the CUDA warp does it (lane k takes call k,
/// k+32, ...), and the row is finished by a subgroup reduction.  Giving one work-item the whole row instead
/// costs a factor of ten: 640 serial dependent iterations cannot be hidden, and only 1024 work-items exist
/// for 3840 lanes.
std::string row_kernel(const char* dot_expr, int wg, int rows_per_group) {
    std::string s;
    s += "#define QK 2560\n";
    s += "#define ROW_WG " + std::to_string(wg) + "\n";
    s += "#define ROWS_PER_GROUP " + std::to_string(rows_per_group) + "\n";
    s += R"CLC(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel __attribute__((reqd_work_group_size(ROW_WG, 1, 1)))
void row(__global const uint* w, __global const uint* x, __global float* out,
         __global const float* dw, __global const float* dx, const uint n_rows) {
    const uint lane = get_local_id(0);
    const uint per = QK / 4;
    const uint base = get_group_id(0) * ROWS_PER_GROUP;
    for (uint r = 0; r < ROWS_PER_GROUP; ++r) {
        const uint row = base + r;
        if (row >= n_rows) return;
        __global const uint* wr = w + (size_t) row * per;
        int acc = 0;
        for (uint k = lane; k < per; k += ROW_WG) {
            const uint xa = x[k], wa = wr[k];
            acc = DOTEXPR;
        }
        const int total = sub_group_reduce_add(acc);
        if (lane == 0) out[row] = (float) total * dw[row] * dx[0];
    }
}
)CLC";
    // substitute last, so the macro name never appears in the emitted source by accident
    const std::string key = "DOTEXPR";
    for (size_t at = s.find(key); at != std::string::npos; at = s.find(key, at + 1))
        s.replace(at, key.size(), dot_expr);
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const int N = 1 << 16;
    const unsigned N_ROWS = 4096u;
    const unsigned QK_W = 2560u;
    const int repeat = argc > 1 ? std::atoi(argv[1]) : 8;
    const int rows_per_item = argc > 2 ? std::atoi(argv[2]) : 4;

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
        if (std::strstr(v.data(), "Advanced Micro")) plat = plats[i];
    }
    if (!plat) { std::printf("no AMD platform\n"); return 1; }

    cl_uint nd = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    cl_device_id dev = devs[0];

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &rc);
    if (rc != CL_SUCCESS) { std::printf("queue: %s\n", err_name(rc)); return 1; }

    // ---- 1. which dot forms exist -------------------------------------------------------------------
    std::printf("dot candidates (one program each, so a rejection cannot mask the rest):\n");
    std::vector<cl_program> progs;
    std::vector<std::string> names;
    // amd_mqsad is deliberately absent.  It is present in the driver (and in amdhsail64.dll), but building
    // anything that calls it aborts the process inside the driver:
    //     Error in hsa_operand section, at offset 948: Invalid operand 3, expected: a register
    //     LLVM ERROR: Brig container validation has failed in BRIGAsmPrinter.cpp
    // A library that can kill the host process is not a building block, so the two portable spellings below
    // are the ones the expert kernels will use.
    struct { const char* label; const char* src; } cands[] = {
        {"dot_acc_sat_4x8packed_ss_int", kDotOcl20},
        {"mad24 x4", kDotMad24},
        {"plain signed mul x4", kDotPlain},
    };
    for (auto& c : cands) {
        bool ok = false;
        cl_program p = build_one(ctx, dev, c.label, c.src, ok);
        if (ok) { progs.push_back(p); names.push_back(c.label); }
    }

    // ---- 2. semantics ------------------------------------------------------------------------------
    std::vector<uint32_t> ha(N), hb(N);
    std::mt19937 rng(0xC0FFEE);
    for (int i = 0; i < N; ++i) { ha[i] = rng(); hb[i] = rng(); }
    ha[0] = 0x7F7F7F7Fu; hb[0] = 0x7F7F7F7Fu;   // +127 * +127 * 4
    ha[1] = 0x80808080u; hb[1] = 0x80808080u;   // -128 * -128 * 4
    ha[2] = 0xFFFFFF7Fu; hb[2] = 0x00000080u;

    cl_mem ba = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t) N * 4, ha.data(), &rc);
    cl_mem bb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t) N * 4, hb.data(), &rc);
    cl_mem bo = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) N * 4, nullptr, &rc);
    std::vector<int32_t> ho(N, 0);

    std::printf("\n4x8 dot semantics against a host reference:\n");
    for (size_t i = 0; i < progs.size(); ++i) {
        cl_kernel k = clCreateKernel(progs[i], "d", &rc);
        if (rc != CL_SUCCESS) { std::printf("  %-32s kernel 'd' missing\n", names[i].c_str()); continue; }
        clSetKernelArg(k, 0, sizeof(ba), &ba);
        clSetKernelArg(k, 1, sizeof(bb), &bb);
        clSetKernelArg(k, 2, sizeof(bo), &bo);
        const size_t g = (size_t) N;
        rc = clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
        if (rc != CL_SUCCESS) {
            std::printf("  %-32s enqueue %s\n", names[i].c_str(), err_name(rc));
            clReleaseKernel(k);
            continue;
        }
        clFinish(q);
        clEnqueueReadBuffer(q, bo, CL_TRUE, 0, (size_t) N * 4, ho.data(), 0, nullptr, nullptr);
        int bad = 0, first = -1;
        for (int j = 0; j < N; ++j)
            if (ho[j] != dot4_ref(ha[j], hb[j])) { if (first < 0) first = j; ++bad; }
        if (bad == 0)
            std::printf("  %-32s EXACT (0/%d)\n", names[i].c_str(), N);
        else
            std::printf("  %-32s DIFFERS on %d/%d  (first j=%d: got %d, true dot4 = %d)\n",
                        names[i].c_str(), bad, N, first, ho[first], dot4_ref(ha[first], hb[first]));
        clReleaseKernel(k);
    }

    // ---- 3. row kernel bandwidth --------------------------------------------------------------------
    const size_t per = QK_W / 4;
    const size_t w_count = (size_t) N_ROWS * per;
    std::vector<uint32_t> hw(w_count);
    for (size_t i = 0; i < w_count; ++i) hw[i] = rng();
    std::vector<uint32_t> hx(per);
    for (size_t i = 0; i < per; ++i) hx[i] = rng();
    std::vector<float> hdw(N_ROWS), hdx(1, 1.0f), hout(N_ROWS);
    for (unsigned i = 0; i < N_ROWS; ++i) hdw[i] = 0.01f;

    const double w_bytes = (double) w_count * 4.0;
    std::printf("\nrow kernel: %u rows x %u signed bytes = %.2f MB, repeat %d\n",
                N_ROWS, QK_W, w_bytes / 1e6, repeat);

    cl_mem bw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_bytes, hw.data(), &rc);
    cl_mem bx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, per * 4, hx.data(), &rc);
    cl_mem bdw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N_ROWS * 4, hdw.data(), &rc);
    cl_mem bdx = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, hdx.data(), &rc);
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N_ROWS * 4, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("weight buffers: %s (%.1f MB)\n", err_name(rc), w_bytes / 1e6); return 1; }

    struct Variant { const char* dot; const char* label; int wg; int r; };
    const char* kMad24 = "(int) mad24((int) ((wa << 24) >> 24), (int) ((xa << 24) >> 24), acc)";
    const char* kPlain = "acc + (int) ((wa << 24) >> 24) * (int) ((xa << 24) >> 24)";
    const Variant variants[] = {
        {kMad24, "mad24 x4", 32, 1},
        {kMad24, "mad24 x4", 32, 4},
        {kMad24, "mad24 x4", 64, 1},
        {kMad24, "mad24 x4", 64, 4},
        {kMad24, "mad24 x4", 64, 8},
        {kPlain, "plain mul x4", 32, 4},
        {kPlain, "plain mul x4", 64, 4},
    };
    for (const Variant& v : variants) {
        const std::string src = row_kernel(v.dot, v.wg, v.r);
        const char* srcp = src.c_str();
        cl_int crc = CL_SUCCESS;
        cl_program p = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &crc);
        if (crc != CL_SUCCESS) { std::printf("  %-16s wg=%-3d r=%-2d  source %s\n", v.label, v.wg, v.r, err_name(crc)); continue; }
        crc = clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
        if (crc != CL_SUCCESS) {
            std::printf("  %-16s wg=%-3d r=%-2d  build %s\n", v.label, v.wg, v.r, err_name(crc));
            size_t n = 0;
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n + 1, 0);
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            std::printf("      %s\n", log.data());
            clReleaseProgram(p);
            continue;
        }
        cl_kernel k = clCreateKernel(p, "row", &crc);
        if (crc != CL_SUCCESS) { std::printf("  %-16s wg=%-3d r=%-2d  kernel %s\n", v.label, v.wg, v.r, err_name(crc)); clReleaseProgram(p); continue; }
        clSetKernelArg(k, 0, sizeof(bw), &bw);
        clSetKernelArg(k, 1, sizeof(bx), &bx);
        clSetKernelArg(k, 2, sizeof(bout), &bout);
        clSetKernelArg(k, 3, sizeof(bdw), &bdw);
        clSetKernelArg(k, 4, sizeof(bdx), &bdx);
        clSetKernelArg(k, 5, sizeof(N_ROWS), &N_ROWS);

        const size_t groups = (N_ROWS + (size_t) v.r - 1) / (size_t) v.r;
        const size_t g[1] = {groups * (size_t) v.wg};
        const size_t l[1] = {(size_t) v.wg};
        cl_int erc = CL_SUCCESS;
        for (int i = 0; i < 3; ++i) {
            erc = clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
            if (erc != CL_SUCCESS) break;
        }
        if (erc != CL_SUCCESS) {
            std::printf("  %-16s wg=%-3d r=%-2d  enqueue %s\n", v.label, v.wg, v.r, err_name(erc));
            clReleaseKernel(k);
            clReleaseProgram(p);
            continue;
        }
        clFinish(q);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i)
            clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        clFinish(q);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
        const double gbs = w_bytes / (ms * 1e-3) / 1e9;
        const double mrows = (double) N_ROWS / (ms * 1e-3) / 1e6;
        std::printf("  %-16s wg=%-3d r=%-2d  %7.3f ms  %6.1f GB/s  (%6.1f M rows/s)\n",
                    v.label, v.wg, v.r, ms, gbs, mrows);
        clReleaseKernel(k);
        clReleaseProgram(p);
    }

    std::printf("\ndone\n");
    return 0;
}
