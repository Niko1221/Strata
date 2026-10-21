// maxwell-instr-rates.cu - the instruction rates of this card, for the trellis kernels' sake.
//
// The Gyro expert kernels (native_gu_multi_kernel<145>, native_down_multi_kernel<146>) are integer-bound on
// this card: cuobjdump shows them at 160-176 VMAD (the dp4a fallback), 245-312 XMAD (IMAD), 240-254 IADD and
// only 10-14 FFMA per kernel body, and native_expert_bench measures them at 18 GB/s against the card's ~336
// GB/s of memory bandwidth.  So: which pipe is the limit?  This measures the throughput of the instructions
// the kernel is made of, each on its own, so the answer is a number and not a theory.
//
//   nvcc -O3 -arch=sm_52 -ccbin /usr/bin/g++-14 maxwell-instr-rates.cu -o maxwell-instr-rates
//   ./maxwell-instr-rates
//
// The reported rate is the instructions per clock per SM: the kernel does a fixed number of them in a loop
// with a known cycle count (the %clock64 counter needs the counters ncu cannot read here, so the rate comes
// from the wall time against the SM's clock, which the driver reports).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define ITERS 4096
#define BLOCKS 1024
#define THREADS 256

__device__ __forceinline__ int vmad_dp4a(int a, int b, int c) {   // the engine's sm_50 fallback: 4 vmad
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;" : "+r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ int byte_dp4a(int a, int b, int c) {     // the byte-wise reference
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
__device__ __forceinline__ int idp4a(int a, int b, int c) {        // what sm_61+ has
    int r = c;
    asm("dp4a.s32.s32.s32 %0, %1, %2, %0;" : "+r"(r) : "r"(a), "r"(b));
    return r;
}
#else
__device__ __forceinline__ int idp4a(int a, int b, int c) {        // not on this card: the instruction
    return byte_dp4a(a, b, c);                                     // does not exist below sm_61
}
#endif

__global__ void k_vmad(int* out, int iters) {
    int a = threadIdx.x * 0x01020304, b = blockIdx.x * 0x05060708 + 1, acc = 0;
    for (int i = 0; i < iters; ++i) { acc = vmad_dp4a(a, b, acc); a ^= acc; b += 0x01010101; }
    if (acc == 0x7fffffff) out[0] = acc;
}
__global__ void k_idp4a(int* out, int iters) {
    int a = threadIdx.x * 0x01020304, b = blockIdx.x * 0x05060708 + 1, acc = 0;
    for (int i = 0; i < iters; ++i) { acc = idp4a(a, b, acc); a ^= acc; b += 0x01010101; }
    if (acc == 0x7fffffff) out[0] = acc;
}
__global__ void k_byte(int* out, int iters) {
    int a = threadIdx.x * 0x01020304, b = blockIdx.x * 0x05060708 + 1, acc = 0;
    for (int i = 0; i < iters; ++i) { acc = byte_dp4a(a, b, acc); a ^= acc; b += 0x01010101; }
    if (acc == 0x7fffffff) out[0] = acc;
}
__global__ void k_ffma(float* out, int iters) {
    float a = threadIdx.x * 1.000001f, b = blockIdx.x * 1.000003f + 1.0f, acc = 0.0f;
    for (int i = 0; i < iters; ++i) { acc = fmaf(a, b, acc); a = fmaf(acc, 1e-7f, a); b += 1e-6f; }
    if (acc == 12345.678f) out[0] = acc;
}
__global__ void k_imad(int* out, int iters) {
    int a = threadIdx.x * 0x01020304, b = blockIdx.x * 0x05060708 + 1, acc = 0;
    for (int i = 0; i < iters; ++i) { acc = a * b + acc; a ^= acc; b += 0x01010101; }
    if (acc == 0x7fffffff) out[0] = acc;
}
__global__ void k_f2f(float* out, int iters) {   // the trellis codebook's F16 -> F32
    unsigned short h = (unsigned short) (threadIdx.x * 977);
    float acc = 0.0f;
    for (int i = 0; i < iters; ++i) { acc += __half2float(__ushort_as_half(h)); h = (unsigned short) (h * 3 + 1); }
    if (acc == 12345.678f) out[0] = acc;
}

struct Dev { int sm, clock_mhz; };

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    Dev d{p.multiProcessorCount, p.clockRate / 1000};
    printf("device: %s (sm_%d%d), %d SMs, clock %.2f GHz\n", p.name, p.major, p.minor, d.sm, d.clock_mhz / 1000.0);
    int* out = nullptr;
    cudaMalloc(&out, 4);
    const long long threads = (long long) BLOCKS * THREADS;
    const long long ops = threads * ITERS;
    struct Row { const char* name; int per_iter; };
    const Row rows[] = {{"vmad x4 (the sm_50 dp4a fallback)", 4}, {"idp4a (sm_61+)", 1}, {"byte-wise dp4a", 4},
                        {"ffma", 1}, {"imad", 1}, {"f2f.f16->f32", 1}};
    for (int r = 0; r < 6; ++r) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int w = 0; w < 2; ++w) {   // a warm pass, then the measured one
            switch (r) {
                case 0: k_vmad<<<BLOCKS, THREADS>>>(out, ITERS); break;
                case 1: k_idp4a<<<BLOCKS, THREADS>>>(out, ITERS); break;
                case 2: k_byte<<<BLOCKS, THREADS>>>(out, ITERS); break;
                case 3: k_ffma<<<BLOCKS, THREADS>>>((float*) out, ITERS); break;
                case 4: k_imad<<<BLOCKS, THREADS>>>(out, ITERS); break;
                default: k_f2f<<<BLOCKS, THREADS>>>((float*) out, ITERS); break;
            }
        }
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        switch (r) {
            case 0: k_vmad<<<BLOCKS, THREADS>>>(out, ITERS); break;
            case 1: k_idp4a<<<BLOCKS, THREADS>>>(out, ITERS); break;
            case 2: k_byte<<<BLOCKS, THREADS>>>(out, ITERS); break;
            case 3: k_ffma<<<BLOCKS, THREADS>>>((float*) out, ITERS); break;
            case 4: k_imad<<<BLOCKS, THREADS>>>(out, ITERS); break;
            default: k_f2f<<<BLOCKS, THREADS>>>((float*) out, ITERS); break;
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double clocks = (double) d.clock_mhz * 1e6 * (ms / 1000.0);   // the SM clocks in the interval
        const double instr = ops * rows[r].per_iter;
        printf("%-34s %8.2f ms   %7.2f G instr/s   %6.1f instr/SM/clk\n", rows[r].name, ms, instr / (ms / 1000.0) / 1e9,
               instr / clocks);
    }
    cudaFree(out);
    printf("cuda: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
