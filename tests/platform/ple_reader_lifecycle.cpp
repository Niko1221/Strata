// CPU-only lifecycle regressions using real DirectFile I/O; allocation failure is injected at link time.
#include "strata/ngram/ple_reader.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef STRATA_TEST_ALLOC_FAILURE
static bool fail_alloc = false;
extern "C" int __real_posix_memalign(void**, size_t, size_t);
extern "C" int __wrap_posix_memalign(void** out, size_t alignment, size_t bytes) {
    if (fail_alloc) { fail_alloc = false; return ENOMEM; }
    return __real_posix_memalign(out, alignment, bytes);
}
#endif

namespace {
using strata::ngram::PleReader;
constexpr uint32_t kRows = 128, kRowBytes = 90;
constexpr size_t kFileBytes = 16384;
void check(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
uint8_t pattern(size_t i) { return (uint8_t) ((i * 17 + i / 257) & 255); }

struct Table {
    std::filesystem::path path;
    Table() {
        std::string name = (std::filesystem::temp_directory_path() / "strata-ple-lifecycle-XXXXXX").string();
        const int fd = mkstemp(name.data());
        check(fd >= 0, "mkstemp failed");
        ::close(fd);
        path = name;
        std::vector<uint8_t> bytes(kFileBytes);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = pattern(i);
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
        check((bool) f, "cannot write fixture");
    }
    ~Table() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};

void open(PleReader& r, const Table& table, bool threaded) {
    std::string err;
    const bool ok = r.open(table.path.string(), 0, kRows, 4, 0, err, threaded, kRowBytes);
    check(ok, "open failed: " + err);
}

void read(PleReader& r) {
    const uint32_t rows[] = {0, 45, 91, 127}; // different pages, including a row that crosses a page boundary
    std::vector<uint8_t> out(4 * kRowBytes, 0);
    const auto ticket = r.issue(rows, 4, out.data());
    std::string err;
    const bool ok = r.collect(ticket, err);
    check(ok, "collect failed: " + err);
    for (size_t i = 0; i < 4; ++i)
        for (size_t b = 0; b < kRowBytes; ++b)
            check(out[i * kRowBytes + b] == pattern((size_t) rows[i] * kRowBytes + b), "row bytes changed");
}

void refused(PleReader& r, const Table& table, bool threaded, bool allocation) {
    std::string err;
#ifdef STRATA_TEST_ALLOC_FAILURE
    if (allocation) fail_alloc = true;
#endif
    const bool ok = r.open(table.path.string(), allocation ? 0 : kFileBytes, kRows, 4, 0, err,
                           threaded, kRowBytes);
    check(!ok, "invalid initialization was accepted");
    check(err.find(allocation ? "cannot allocate read buffers" : "table extends past the end") != std::string::npos,
          "unexpected initialization error: " + err);
#ifdef STRATA_TEST_ALLOC_FAILURE
    check(!fail_alloc, "allocation failure was not exercised");
#endif
    check(!r.is_open(), "failed reader kept its file open");
    r.close();
    r.close();
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 3, "usage: ple_reader_lifecycle_test reads|reopen|alloc caller|threaded");
        const std::string mode = argv[1];
        const bool threaded = std::string(argv[2]) == "threaded";
        check(mode == "reads" || mode == "reopen" || mode == "alloc", "unknown case");
        Table table;
        PleReader r;
        if (mode != "reads") refused(r, table, threaded, mode == "alloc");
        open(r, table, threaded);
        read(r);
        // open() must also clean up a previously initialized reader before handling the new failure.
        if (mode != "reads") refused(r, table, threaded, mode == "alloc");
        else {
            const uint32_t rows[] = {0, 45, 91, 127};
            std::vector<uint8_t> out(4 * kRowBytes, 0);
            (void) r.issue(rows, 4, out.data());
            r.close(); // outstanding writes must finish before out is released
        }
        r.close();
        open(r, table, threaded);
        read(r);
        r.close();
        r.close();
        std::printf("%s %s: passed\n", argv[1], argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
