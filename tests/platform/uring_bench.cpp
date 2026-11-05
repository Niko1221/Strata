// Standalone, opt-in I/O throughput benchmark: same aligned blocks via DirectFile or blocking pread.
// Run STRATA_IO_URING=1 strata_uring_bench to compare; gracefully reports a fallback on seccomp kernels.
#include "strata/platform/direct_file.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

int main() {
    using strata::platform::DirectFile;
    constexpr int blocks = 256, bytes = 4096;
    char path[] = "/tmp/strata-uring-bench-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    std::vector<unsigned char> data(blocks * bytes);
    for (size_t i = 0; i < data.size(); ++i) data[i] = (unsigned char) (i * 19u + i / bytes);
    if (write(fd, data.data(), data.size()) != (ssize_t) data.size() || fsync(fd) != 0) return 2;
    close(fd);
    DirectFile file;
    std::string err;
    if (!file.open(path, err)) { std::fprintf(stderr, "%s\n", err.c_str()); unlink(path); return 3; }
    std::vector<void*> buffers(blocks);
    for (void*& p : buffers) { p = DirectFile::alloc_aligned(bytes); if (!p) return 4; }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < blocks; ++i)
        if (!file.submit((uint64_t) i * bytes, buffers[i], bytes, (uint64_t) i, err)) return 5;
    bool seen[blocks] = {};
    int got = 0;
    while (got < blocks) {
        strata::platform::Completion c[64];
        const int n = file.wait(c, 64, 5000);
        if (n == 0) return 6;
        for (int i = 0; i < n; ++i) {
            if (!c[i].ok || c[i].bytes != bytes || c[i].tag >= blocks || seen[c[i].tag]) return 7;
            if (std::memcmp(buffers[c[i].tag], data.data() + c[i].tag * bytes, bytes) != 0) return 8;
            seen[c[i].tag] = true;
            ++got;
        }
    }
    const double async_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const bool ring = file.using_uring();
    file.close();
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return 9;
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < blocks; ++i)
        if (pread(fd, buffers[i], bytes, (off_t) i * bytes) != bytes) return 10;
    const double sync_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    close(fd);
    for (void* p : buffers) DirectFile::free_aligned(p);
    unlink(path);
    std::printf("backend=%s  asynchronous=%.2f MiB/s  blocking pread=%.2f MiB/s (O_DIRECT fixture; not an SSD benchmark)\n",
                ring ? "io_uring" : "pread workers", (double) blocks * bytes / async_s / 1048576.0,
                (double) blocks * bytes / sync_s / 1048576.0);
    return 0;
}