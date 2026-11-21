// Which earlier call makes the next hipStreamCreateWithFlags crash on ROCm 10.1?
//   ./narrow <steps>   steps: any of  H=hipHostMalloc 256 MiB  M=hipMalloc 256 MiB  m=hipMalloc 4 KiB
//                      C=hipMemcpy H2D 4 KiB from a stack buffer  S=hipStreamCreateWithFlags(NonBlocking)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
    const char* steps = argc > 1 ? argv[1] : "S";
    void* h = nullptr; void* d = nullptr; void* small = nullptr; char buf[4096] = {};
    for (const char* c = steps; *c; ++c) {
        std::printf("step %c\n", *c); std::fflush(stdout);
        hipError_t e = hipSuccess;
        switch (*c) {
            case 'H': e = hipHostMalloc(&h, 256u << 20, hipHostMallocDefault); break;
            case 'M': e = hipMalloc(&d, 256u << 20); break;
            case 'm': e = hipMalloc(&small, 4096); break;
            case 'C': if (!small) hipMalloc(&small, 4096); e = hipMemcpy(small, buf, sizeof buf, hipMemcpyHostToDevice); break;
            case 'S': { hipStream_t s; e = hipStreamCreateWithFlags(&s, hipStreamNonBlocking); break; }
        }
        if (e != hipSuccess) { std::printf("step %c: %s\n", *c, hipGetErrorString(e)); return 1; }
    }
    std::printf("OK\n"); return 0;
}
