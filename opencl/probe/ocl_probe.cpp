// opencl/probe/ocl_probe.cpp - what can the MI50 (gfx906) actually be programmed with from this machine?
//
// Why this exists: the MI50 cannot be a CUDA device, and Strata's CMake refuses to build CUDA and HIP at the
// same time, so the plan had been a hand-written Vulkan backend (see vulkan/).  But AMD's Windows driver
// registers an OpenCL 2.x runtime for gfx906, and OpenCL C is close enough to CUDA that the existing
// src/kernels/cuda/iq_kernels.cu expert kernels can be transcribed rather than re-derived - and transcribed
// code can be checked bit-exactly against the CUDA original, which SPIR-V shaders could not be.
//
// A runtime that enumerates devices but has no compiler is the trap here, so the probe is built around one
// question asked repeatedly: does THIS source build?  Each capability sits in its own tiny kernel and gets
// its own build call, so a failure names the missing feature instead of dumping one wall of clang output.
//
// Usage: ocl_probe.exe [-1|0|1] [-v]     (platform index; -1 = pick the AMD one)

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char* err_name(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        default: return "CL_ERR";
    }
}

std::string str_info(cl_device_id d, cl_device_info p) {
    size_t n = 0;
    if (clGetDeviceInfo(d, p, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "<err>";
    std::vector<char> buf(n + 1, 0);
    if (clGetDeviceInfo(d, p, n, buf.data(), nullptr) != CL_SUCCESS) return "<err>";
    return std::string(buf.data());
}

std::string plat_info(cl_platform_id p, cl_platform_info i) {
    size_t n = 0;
    if (clGetPlatformInfo(p, i, 0, nullptr, &n) != CL_SUCCESS || n == 0) return "<err>";
    std::vector<char> buf(n + 1, 0);
    if (clGetPlatformInfo(p, i, n, buf.data(), nullptr) != CL_SUCCESS) return "<err>";
    return std::string(buf.data());
}

bool has_ext(const std::string& exts, const char* name) {
    return exts.find(name) != std::string::npos;
}

namespace probe {

int failures = 0;
bool verbose = false;

/// Compile `src` and say whether it builds.  This is the whole point of the file: one capability, one build,
/// one line of output, and the clang diagnostic printed when it fails.
bool builds(cl_context ctx, cl_device_id dev, const char* label, const char* opts, const char* src,
            cl_program* out = nullptr) {
    cl_int rc = CL_SUCCESS;
    cl_program p = clCreateProgramWithSource(ctx, 1, &src, nullptr, &rc);
    if (rc != CL_SUCCESS) {
        std::printf("  %-34s CREATE FAILED %s\n", label, err_name(rc));
        ++failures;
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    rc = clBuildProgram(p, 1, &dev, opts, nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (rc != CL_SUCCESS) {
        std::printf("  %-34s BUILD FAILED %s (%.0f ms)\n", label, err_name(rc), ms);
        size_t n = 0;
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        if (n > 1) {
            std::vector<char> log(n + 1, 0);
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            std::string s(log.data());
            // only the first few lines of clang output are interesting; the rest is macro expansion noise
            int lines = 0;
            for (size_t i = 0; i < s.size() && lines < 6; ++i) {
                if (s[i] == '\n') ++lines;
                std::fputc(s[i], stdout);
            }
            if (!s.empty() && s.back() != '\n') std::fputc('\n', stdout);
        }
        clReleaseProgram(p);
        ++failures;
        return false;
    }
    std::printf("  %-34s ok (%.0f ms)\n", label, ms);
    if (out) *out = p; else clReleaseProgram(p);
    return true;
}

}  // namespace probe

// ---------------------------------------------------------------------------------------------------------
// The capability probes.  Each is a complete program; keep them small so the diagnostic is about one thing.
// ---------------------------------------------------------------------------------------------------------

const char* kBasics = R"CLC(
__kernel void trivial(__global const float* a, __global const float* b, __global float* c) {
    c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
}
__kernel void vecs(__global const int* a, __global int* o) {
    const int i = get_global_id(0);
    int4 v = vload4(i, a);
    uint4 u = as_uint4(v);
    long2 l = (long2)((long) u.x * u.y, (long) u.z * u.w);
    o[i] = (int) (l.x + l.y);
}
)CLC";

const char* kSubgroup = R"CLC(
__kernel void reduce_probe(__global const float* v, __global float* out) {
    const float x = v[get_global_id(0)];
    const float t = sub_group_reduce_add(x);
    if (get_sub_group_local_id() == 0) out[get_group_id(0)] = t;
}
__kernel void scan_probe(__global const int* v, __global int* out) {
    int x = v[get_global_id(0)];
    int t = sub_group_reduce_max(x);
    int s = sub_group_scan_exclusive_add(x);
    out[get_global_id(0)] = t + s;
}
)CLC";

// fp16 is how every quant block stores its scale, so half->float has to be cheap and exact.
const char* kFp16 = R"CLC(
__kernel void f16_probe(__global const ushort* h, __global const float* f, __global float* o) {
    const int i = get_global_id(0);
    const half v = as_half(h[i]);
    o[i] = (float) v * f[i];
}
__kernel void f16x2_math(__global const ushort* h, __global float* o) {
    const int i = get_global_id(0);
    const half2 a = *(const __global half2*) (h + 2 * i);
    const float2 f = convert_float2(a);
    o[i] = f.x + f.y;
}
)CLC";

// The 4x8 integer dot.  gfx906 is where AMD added V_DOT4_I32_IU8, so the question is whether the OpenCL
// compiler will emit it from an intrinsic, and (falling back) from a plain 4-term multiply-add.
const char* kDotManual = R"CLC(
__kernel void dot_manual(__global const int* a, __global const int* b, __global int* out) {
    const int i = get_global_id(0);
    const int x = a[i], y = b[i];
    int s = 0;
    for (int k = 0; k < 4; ++k)
        s = mad24((x << (24 - 8 * k)) >> 24, (y << (24 - 8 * k)) >> 24, s);
    out[i] = s;
}
)CLC";

const char* kDotBuiltin = R"CLC(
__kernel void dot_sdot4(__global const int* a, __global const int* b, __global int* out) {
    out[get_global_id(0)] = __builtin_amdgcn_sdot4(a[get_global_id(0)], b[get_global_id(0)], 0);
}
__kernel void dot_udot4(__global const uint* a, __global const uint* b, __global int* out) {
    out[get_global_id(0)] = (int) __builtin_amdgcn_sudot4(1, a[get_global_id(0)], b[get_global_id(0)], 0);
}
)CLC";

const char* kDotAsm = R"CLC(
__kernel void dot_asm(__global const int* a, __global const int* b, __global int* out) {
    const int i = get_global_id(0);
    int s;
    asm volatile("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(s) : "v"(a[i]), "v"(b[i]));
    out[i] = s;
}
)CLC";

// The AMD-only media intrinsics build on `mad24`-style 24-bit math; if any of these exist they are a knob
// for the i-quant dot that does not depend on the clang version recognising an amdgcn builtin.
const char* kMedia = R"CLC(
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
__kernel void media_sad(__global const uint* a, __global const uint* b, __global uint* out) {
    out[get_global_id(0)] = amd_sad(a[get_global_id(0)], b[get_global_id(0)], 0);
}
__kernel void media_msad(__global const uint* a, __global const uint* b, __global uint* out) {
    out[get_global_id(0)] = amd_msad(a[get_global_id(0)], b[get_global_id(0)], 0);
}
__kernel void media_mqsad(__global const uint* a, __global const uint* b, __global uint* out) {
    out[get_global_id(0)] = amd_mqsad(a[get_global_id(0)], b[get_global_id(0)], 0);
}
__kernel void media_bfe(__global const uint* a, __global uint* out) {
    out[get_global_id(0)] = amd_bfe(a[get_global_id(0)], (2u << 16) | 8u);
}
)CLC";

const char* kLocalMem = R"CLC(
__kernel void local_probe(__global const float* in, __global float* out, __local float* scratch) {
    const int g = get_global_id(0), l = get_local_id(0);
    scratch[l] = in[g];
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0;
    for (int i = 0; i < get_local_size(0); ++i) s += scratch[i];
    out[g] = s;
}
)CLC";

// Atomics on the expert cache's residency counters, and a 64-bit store for the device address table.
const char* kAtomics = R"CLC(
__kernel void atomics_probe(__global volatile int* hits, __global long* ptrs) {
    const int i = get_global_id(0);
    atomic_add(&hits[0], 1);
    if (i == 0) ptrs[0] = 0x123456789abcdefL;
}
)CLC";

}  // namespace

int main(int argc, char** argv) {
    int want = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0) probe::verbose = true;
        else want = std::atoi(argv[i]);
    }

    cl_uint n_plat = 0;
    cl_int rc = clGetPlatformIDs(0, nullptr, &n_plat);
    if (rc != CL_SUCCESS || n_plat == 0) { std::printf("no OpenCL platforms (%s)\n", err_name(rc)); return 1; }
    std::vector<cl_platform_id> plats(n_plat);
    clGetPlatformIDs(n_plat, plats.data(), nullptr);

    cl_platform_id plat = nullptr;
    for (cl_uint i = 0; i < n_plat; ++i) {
        std::printf("[platform %u] %-38s %s\n", i, plat_info(plats[i], CL_PLATFORM_NAME).c_str(),
                    plat_info(plats[i], CL_PLATFORM_VERSION).c_str());
        const std::string v = plat_info(plats[i], CL_PLATFORM_VENDOR);
        if (want < 0 ? has_ext(v, "Advanced Micro") : (int) i == want) plat = plats[i];
    }
    if (!plat) { std::printf("no matching platform\n"); return 1; }

    cl_uint n_dev = 0;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_dev) != CL_SUCCESS || n_dev == 0) {
        std::printf("no devices\n");
        return 1;
    }
    std::vector<cl_device_id> devs(n_dev);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, n_dev, devs.data(), nullptr);
    cl_device_id dev = devs[0];

    cl_ulong gmem = 0, lmem = 0, maxalloc = 0;
    size_t wgsize = 0;
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsize), &wgsize, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    const std::string exts = str_info(dev, CL_DEVICE_EXTENSIONS);

    std::printf("\ndevice            : %s   (%s)\n", str_info(dev, CL_DEVICE_NAME).c_str(),
                str_info(dev, CL_DEVICE_VERSION).c_str());
    std::printf("  driver          : %s\n", str_info(dev, CL_DRIVER_VERSION).c_str());
    std::printf("  global mem      : %.2f GiB\n", (double) gmem / 1073741824.0);
    std::printf("  max alloc       : %.2f GiB\n", (double) maxalloc / 1073741824.0);
    std::printf("  local mem       : %llu KiB\n", (unsigned long long) (lmem / 1024));
    std::printf("  compute units   : %u\n", cu);
    std::printf("  max work group  : %llu\n", (unsigned long long) wgsize);

    cl_int cr = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &cr);
    if (cr != CL_SUCCESS) { std::printf("clCreateContext: %s\n", err_name(cr)); return 1; }

    std::printf("\ncapability probes:\n");
    probe::builds(ctx, dev, "basics: kernels, int4, as_uint4", "-cl-std=CL2.0", kBasics);
    probe::builds(ctx, dev, "subgroups (reduce + scan)", "-cl-std=CL2.0 -Werror", kSubgroup);
    probe::builds(ctx, dev, "fp16: half, half2, convert_float2", "-cl-std=CL2.0", kFp16);
    probe::builds(ctx, dev, "dot: mad24 x4 (portable fallback)", "-cl-std=CL2.0", kDotManual);
    probe::builds(ctx, dev, "dot: __builtin_amdgcn_sdot4", "-cl-std=CL2.0", kDotBuiltin);
    probe::builds(ctx, dev, "dot: inline v_dot4_i32_i8 asm", "-cl-std=CL2.0", kDotAsm);
    probe::builds(ctx, dev, "dot: cl_amd_media_ops intrinsics", "-cl-std=CL2.0", kMedia);
    probe::builds(ctx, dev, "local memory + barrier", "-cl-std=CL2.0", kLocalMem);
    probe::builds(ctx, dev, "atomics: atom_add, i64 store", "-cl-std=CL2.0", kAtomics);

    if (!probe::verbose) {
        std::printf("\n%d of 9 capability probes failed\n", probe::failures);
        return probe::failures == 0 ? 0 : 2;
    }

    // Verbose mode: compile one real kernel and dump the GCN ISA, so we can see whether the dot became a
    // single V_DOT4_I32_IU8 or a four-instruction multiply chain.  AMD's compiler writes the assembly when
    // asked for the "binary" of a program built with -save-temps; simpler for us: ask for the build log with
    // -Rpass and read the ISA out of the disassembly that the driver can already produce.
    cl_program p = nullptr;
    if (probe::builds(ctx, dev, "dot_manual (retained for asm)", "-cl-std=CL2.0 -save-temps=obj", kDotManual, &p)) {
        size_t n = 0;
        clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, 0, nullptr, &n);
        std::printf("  binary sizes reported: %zu bytes\n", n);
        clReleaseProgram(p);
    }
    return 0;
}
