// vram_hog.cu: holds N MiB of one GPU's memory for S seconds, so a big card behaves like a smaller one (the grid's VRAM-class cells).
// usage: vram_hog <device> <MiB> <seconds>        build: nvcc -O1 vram_hog.cu -o vram_hog
// It prints "hog holds <MiB> MiB" once the memory is held and exits when the time is up or when it is stopped (SIGTERM); cell_runner.py stops only its own hog.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <device> <MiB> <seconds>\n", argv[0]); return 2; }
    const int dev = std::atoi(argv[1]);
    const size_t mib = (size_t)std::atoll(argv[2]);
    const int secs = std::atoi(argv[3]);
    if (cudaSetDevice(dev) != cudaSuccess) { std::fprintf(stderr, "no device %d\n", dev); return 3; }
    size_t got = 0;
    const size_t piece = 256;                       // 256 MiB pieces: the allocator finds room for them more easily
    while (got < mib) {
        void* p = nullptr;
        const size_t n = (mib - got < piece) ? (mib - got) : piece;
        if (cudaMalloc(&p, n << 20) != cudaSuccess) { std::fprintf(stderr, "cudaMalloc failed after %zu MiB\n", got); return 4; }
        got += n;
    }
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    std::printf("hog holds %zu MiB; card now %zu MiB free of %zu MiB\n", got, f >> 20, t >> 20);
    std::fflush(stdout);
    sleep(secs);
    return 0;
}
