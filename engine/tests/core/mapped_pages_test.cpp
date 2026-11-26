// strata/core/mapped_pages.hpp: releasing a copied-out range of a read-only file mapping drops it from the process's resident set, keeps the
// neighbouring pages, and the data reads back unchanged afterwards. The RSS check is Linux only (/proc/self/status); elsewhere only the data checks run.
#include "strata/core/mapped_pages.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

long rss_kb() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "VmRSS:", 6) == 0) { std::sscanf(line + 6, "%ld", &v); break; }
    std::fclose(f);
    return v;
}
}  // namespace

int main() {
    using strata::core::release_mapped_pages;
    // pure range rules need no file
    char probe[16384];
    CHECK(release_mapped_pages(nullptr, 100) == 0);
    CHECK(release_mapped_pages(probe, 0) == 0);
    CHECK(release_mapped_pages(probe + 1, 100) == 0);   // inside one page: nothing whole to release

#ifndef _WIN32
    const size_t kSize = 64u << 20;
    const std::string path = "mapped_pages_test.bin";
    {
        std::vector<unsigned char> buf(kSize);
        for (size_t i = 0; i < kSize; ++i) buf[i] = static_cast<unsigned char>((i * 2654435761u) >> 24);
        std::FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (!f) return 1;
        std::fwrite(buf.data(), 1, kSize, f);
        std::fclose(f);
    }
    int fd = ::open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    void* m = mmap(nullptr, kSize, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(m != MAP_FAILED);
    const unsigned char* d = static_cast<const unsigned char*>(m);

    volatile unsigned sum = 0;   // volatile: an optimised build must really touch every page
    for (size_t i = 0; i < kSize; i += 4096) sum = sum + d[i];
    const long touched = rss_kb();
    const size_t got = release_mapped_pages(d + 4096 + 100, kSize - 8192);   // unaligned both ends: rounded inward
    CHECK(got >= kSize - 3 * 4096 && got <= kSize);
    const long after = rss_kb();
    if (touched > 0 && after > 0) {
        std::printf("RSS touched %ld kB, after release %ld kB\n", touched, after);
        CHECK(touched - after > static_cast<long>(kSize / 1024) * 8 / 10);   // at least 80% of the 64 MB came back
    }
    // the data is unchanged when read again (the pages fault back in from the page cache)
    bool same = true;
    for (size_t i = 0; i < kSize; i += 4097) if (d[i] != static_cast<unsigned char>((i * 2654435761u) >> 24)) { same = false; break; }
    CHECK(same);
    munmap(m, kSize);
    ::close(fd);
    std::remove(path.c_str());
#endif
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::puts("mapped_pages_test: ok");
    return 0;
}
