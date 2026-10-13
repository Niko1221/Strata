// How much of the MI50's 32 GiB can an OpenCL program actually hold?
//
// `CL_DEVICE_MAX_MEM_ALLOC_SIZE` reports 13.60 GiB on this gfx906, which is what caps the peer tier's single
// arena at 13.01 GiB of experts.  The number is a hint from the WDDM-aware runtime, though, not a promise, and
// it says nothing about the total the device will hand out across several buffers.  This probe answers both
// questions directly:
//
//   * will a single allocation larger than the reported maximum succeed at all?
//   * if the arena is split into several buffers, how many GiB can be resident in total?
//
// The answer decides whether the peer tier stays capped where it is or gets a segmented arena and a second
// helping of experts.  AMD's GPU_MAX_ALLOC_PERCENT / GPU_MAX_HEAP_SIZE are tried by the caller, not here.
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char* err_name(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        default: return "CL error";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const double one_gib = 1073741824.0;

    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        std::printf("no OpenCL platform\n");
        return 2;
    }
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);

    cl_platform_id amd = nullptr;
    for (cl_uint p = 0; p < np; ++p) {
        char nm[256] = {0};
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(nm), nm, nullptr);
        if (std::string(nm).find("AMD") != std::string::npos) amd = plats[p];
    }
    if (amd == nullptr) {
        std::printf("no AMD platform with an OpenCL device\n");
        return 2;
    }

    cl_device_id dev = nullptr;
    if (clGetDeviceIDs(amd, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) {
        std::printf("no AMD GPU device\n");
        return 2;
    }

    cl_ulong max_alloc = 0, global = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global), &global, nullptr);
    // CL_DEVICE_GLOBAL_FREE_MEMORY_AMD (0x4008) is not in the CUDA toolkit's cl_ext.h, but the AMD runtime
    // answers it with {free, total} in bytes and the numbers below are worth having without the header
    size_t free_amd[2] = {0, 0};
    clGetDeviceInfo(dev, (cl_device_info) 0x4008, sizeof(free_amd), free_amd, nullptr);
    std::printf("global %.2f GiB   reported max alloc %.2f GiB   free %.2f GiB\n", (double) global / one_gib,
                (double) max_alloc / one_gib, (double) free_amd[0] / one_gib);

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS) {
        std::printf("context: %s\n", err_name(rc));
        return 1;
    }

    // ---- one allocation, as large as asked: is the reported maximum a wall or a hint?
    std::printf("\nsingle allocation:\n");
    for (double gib : {12.0, 13.5, 14.0, 16.0, 20.0, 24.0, 28.0, 30.0}) {
        const size_t bytes = (size_t) (gib * one_gib);
        cl_int r = CL_SUCCESS;
        cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &r);
        std::printf("  %5.1f GiB -> %-32s\n", gib, r == CL_SUCCESS ? "ok" : err_name(r));
        if (m != nullptr) clReleaseMemObject(m);
        else if (r == CL_OUT_OF_RESOURCES || r == CL_MEM_OBJECT_ALLOCATION_FAILURE) break;
    }

    // ---- several of the reported maximum: how much is there in total, and is it fast?
    //
    // This is the number that matters.  If it comes out near 13.6 GiB the tier is already at the ceiling and
    // the only way past it is a different API; if the device holds much more, the next question is whether the
    // extra is HBM or the driver quietly spilling into system RAM across PCIe.  A trivial streaming kernel
    // answers both: a buffer that reads back at hundreds of GB/s is resident, one that reads back at a few
    // GB/s is not, and the peer tier is worth nothing on the second kind.
    const char* src = R"CLC(
__kernel void touch(__global float* p, const ulong n) {
    const ulong i = get_global_id(0);
    if (i < n) p[i] = p[i] * 1.5f + 1.0f;
}
)CLC";
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &rc);
    cl_kernel touch = nullptr;
    cl_command_queue cq = nullptr;
    if (rc == CL_SUCCESS) rc = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (rc == CL_SUCCESS) touch = clCreateKernel(prog, "touch", &rc);
    if (rc == CL_SUCCESS) cq = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &rc);
    if (rc != CL_SUCCESS || touch == nullptr || cq == nullptr) {
        std::printf("\nthe streaming test could not be built: %s\n", err_name(rc));
        clReleaseContext(ctx);
        return 1;
    }

    std::printf("\n%.2f GiB buffers, each streamed once, until the device says no:\n",
                (double) max_alloc / one_gib);
    {
        std::vector<cl_mem> held;
        double total = 0;
        for (int i = 0; i < 4; ++i) {
            cl_int r = CL_SUCCESS;
            cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t) max_alloc, nullptr, &r);
            if (r != CL_SUCCESS) {
                std::printf("  #%d could not even be created: %s  (total %.2f GiB)\n", i + 1, err_name(r), total);
                break;
            }
            const size_t n = (size_t) max_alloc / sizeof(float);
            clSetKernelArg(touch, 0, sizeof(cl_mem), &m);
            clSetKernelArg(touch, 1, sizeof(cl_ulong), &n);
            const size_t gsz = (n + 255) & ~(size_t) 255;
            cl_event ev = nullptr;
            const cl_int kr = clEnqueueNDRangeKernel(cq, touch, 1, nullptr, &gsz, nullptr, 0, nullptr, &ev);
            if (kr != CL_SUCCESS) {
                std::printf("  #%d could not be touched: %s  (total %.2f GiB)\n", i + 1, err_name(kr), total);
                clReleaseMemObject(m);
                break;
            }
            clWaitForEvents(1, &ev);
            cl_ulong t0 = 0, t1 = 0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
            clReleaseEvent(ev);
            const double sec = (double) (t1 - t0) * 1e-9;
            const double gbs = sec > 0 ? (double) max_alloc * 2.0 / sec / 1e9 : 0.0;   // read + write
            held.push_back(m);
            total += (double) max_alloc / one_gib;
            std::printf("  #%d ok   total %6.2f GiB   streamed at %7.1f GB/s%s\n", i + 1, total, gbs,
                        gbs < 50.0 ? "   <- not resident: the driver is spilling to system RAM" : "");
        }
        std::printf("  a device-writable total of %.2f GiB\n", total);
        for (cl_mem m : held) clReleaseMemObject(m);
    }

    // ---- and the same question the way the tier would use it: many smaller pieces, each measured
    std::printf("\n4.00 GiB buffers, each streamed once, until the device says no:\n");
    {
        std::vector<cl_mem> held;
        const size_t bytes = (size_t) (4.0 * one_gib);
        double total = 0;
        for (int i = 0; i < 10; ++i) {
            cl_int r = CL_SUCCESS;
            cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &r);
            if (r != CL_SUCCESS) {
                std::printf("  stopped after %.2f GiB: %s\n", total, err_name(r));
                break;
            }
            const size_t n = bytes / sizeof(float);
            clSetKernelArg(touch, 0, sizeof(cl_mem), &m);
            clSetKernelArg(touch, 1, sizeof(cl_ulong), &n);
            const size_t gsz = (n + 255) & ~(size_t) 255;
            cl_event ev = nullptr;
            const cl_int kr = clEnqueueNDRangeKernel(cq, touch, 1, nullptr, &gsz, nullptr, 0, nullptr, &ev);
            if (kr != CL_SUCCESS) {
                std::printf("  #%d could not be touched: %s  (total %.2f GiB)\n", i + 1, err_name(kr), total);
                clReleaseMemObject(m);
                break;
            }
            clWaitForEvents(1, &ev);
            cl_ulong t0 = 0, t1 = 0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
            clReleaseEvent(ev);
            const double sec = (double) (t1 - t0) * 1e-9;
            const double gbs = sec > 0 ? (double) bytes * 2.0 / sec / 1e9 : 0.0;
            held.push_back(m);
            total += 4.0;
            std::printf("  #%d ok   total %6.2f GiB   %7.1f GB/s%s\n", i + 1, total, gbs,
                        gbs < 50.0 ? "   <- not resident" : "");
        }
        for (cl_mem m : held) clReleaseMemObject(m);
    }

    clReleaseKernel(touch);
    clReleaseCommandQueue(cq);
    clReleaseProgram(prog);
    clReleaseContext(ctx);
    (void) argc;
    (void) argv;
    return 0;
}
