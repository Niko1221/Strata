// busy_flag_probe: does the spin kernel of ds4_busy_begin (mtl_rt.mm) see the host's write to its shared flag?
//
//   clang++ -std=c++17 -O2 -fobjc-arc busy_flag_probe.mm -framework Metal -framework Foundation -o busy_flag_probe
//   ./busy_flag_probe [cap]
//
// For each variant of the loop's flag read (relaxed load as shipped; a read-modify-write that adds 0; a load behind
// threadgroup_barrier(mem_device)), a spin kernel is committed, the host sets the flag 2 ms later, and the time until
// the command buffer completes is measured.  A kernel that sees the flag leaves within about a millisecond of it; one that
// does not runs to its cap.  The "never set" row gives the cap's own duration.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void spin_load(device volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap; ++i) {
        for (uint j = 0; j < 64; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 16; ++j) f = fma(f, 1.0000001f, 1e-7f);
        x ^= as_type<uint>(f);
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
kernel void spin_rmw(device volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap; ++i) {
        for (uint j = 0; j < 64; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 16; ++j) f = fma(f, 1.0000001f, 1e-7f);
        x ^= as_type<uint>(f);
        if (atomic_fetch_or_explicit(flag, 0u, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
kernel void spin_load_every8(device volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    // the same load, but the loop body is short: the flag is read 8x as often (shows whether the cost is the cache or the loop length)
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap * 8; ++i) {
        for (uint j = 0; j < 8; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 2; ++j) f = fma(f, 1.0000001f, 1e-7f);
        x ^= as_type<uint>(f);
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
kernel void spin_fence(device volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    // a system-scope fence before each read: asks for the host's write to be visible
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap; ++i) {
        for (uint j = 0; j < 64; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 16; ++j) f = fma(f, 1.0000001f, 1e-7f);
        x ^= as_type<uint>(f);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
kernel void spin_sys(device coherent(device) volatile atomic_uint* flag [[buffer(0)]], constant uint& cap [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    // the flag declared coherent(device)
    uint x = t;
    float f = (float) t * 1e-9f;
    for (uint i = 0; i < cap; ++i) {
        for (uint j = 0; j < 64; ++j) x = x * 1664525u + 1013904223u;
        for (uint j = 0; j < 16; ++j) f = fma(f, 1.0000001f, 1e-7f);
        x ^= as_type<uint>(f);
        if (atomic_load_explicit(flag, memory_order_relaxed) != 0u) break;
    }
    if (x == 0xdeadbeefu) atomic_store_explicit(flag, 2u, memory_order_relaxed);
}
)MSL";

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const uint32_t cap = argc > 1 ? (uint32_t) std::atoi(argv[1]) : 20000u;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { std::fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> flag = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        std::printf("device %s, cap %u\n", [[dev name] UTF8String], cap);
        const char* names[] = {"spin_load", "spin_rmw", "spin_fence", "spin_sys"};
        for (const char* nm : names) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:nm]] error:&err];
            for (int mode = 0; mode < 2; ++mode) {   // 0: the host sets the flag after 2 ms; 1: never (the cap's own duration)
                double leave[5];
                for (int rep = 0; rep < 5; ++rep) {
                    *(volatile uint32_t*) [flag contents] = 0;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:flag offset:0 atIndex:0];
                    uint32_t c = cap;
                    [enc setBytes:&c length:4 atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                    const double t0 = now_ms();
                    [cb commit];
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    const double tset = now_ms();
                    if (mode == 0) *(volatile uint32_t*) [flag contents] = 1;
                    [cb waitUntilCompleted];
                    const double t1 = now_ms();
                    leave[rep] = mode == 0 ? t1 - tset : t1 - t0;
                }
                double mn = leave[0], mx = leave[0], sum = 0;
                for (double v : leave) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
                std::printf("%-17s %s: %s  min %.2f  mean %.2f  max %.2f ms\n", nm, mode == 0 ? "flag set at 2 ms " : "flag never set  ",
                            mode == 0 ? "ms from the set to the end" : "ms from commit to the end   ", mn, sum / 5, mx);
            }
        }
    }
    return 0;
}
