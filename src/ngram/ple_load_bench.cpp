// CPU-only measurement of the actual PleTable startup path, including page
// residency and the unchanged Q8 row interpretation. No global cache eviction.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/ngram.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

uint64_t io_bytes() {
    std::ifstream in("/proc/self/io"); std::string key; uint64_t value;
    while (in >> key >> value) if (key == "read_bytes:") return value;
    return 0;
}
uint64_t resident(void* base, size_t bytes, size_t page) {
    std::vector<unsigned char> vec(1 + (bytes - 1) / page);
    if (mincore(base, bytes, vec.data())) throw std::runtime_error("mincore failed");
    uint64_t count = 0; for (auto v : vec) count += (v & 1); return count;
}
int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: ple_load_bench TABLE.gguf cold|warm\n"); return 2; }
    try {
        strata::GgufFile source(argv[1]);
        const auto* tensor = source.find("per_layer_token_embd.weight");
        if (!tensor || tensor->type != 8 || tensor->shape.size() != 2 || tensor->shape[0] != 160)
            throw std::runtime_error("requires validated Q8_0 table");
        const auto* data = source.tensor_data(*tensor);
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        const uint64_t bytes = tensor->shape[1] * 170;
        const uint64_t offset = source.data_start() + tensor->offset;
        if (!page || offset > source.file_size() || bytes > source.file_size() - offset)
            throw std::runtime_error("invalid table bounds");
        const auto begin = (uintptr_t) data & ~(uintptr_t) (page - 1);
        const size_t mapped_bytes = (uintptr_t) data + bytes - begin;
        int advice_error = 0;
        if (std::string(argv[2]) == "cold") {
            // Evict only whole pages contained inside this table. Header and
            // adjacent tensors remain untouched. Verify with mincore below.
            const uint64_t lo = (offset + page - 1) / page * page;
            const uint64_t hi = (offset + bytes) / page * page;
            const int fd = open(argv[1], O_RDONLY);
            if (fd < 0) throw std::runtime_error("open for fadvise failed");
            madvise((void*) begin, mapped_bytes, MADV_DONTNEED);
            advice_error = posix_fadvise(fd, (off_t) lo, (off_t) (hi - lo), POSIX_FADV_DONTNEED);
            close(fd);
        } else if (std::string(argv[2]) != "warm") throw std::runtime_error("invalid cache condition");
        const auto before_pages = resident((void*) begin, mapped_bytes, page);
        const auto read_before = io_bytes();
        rusage before{}, after{}; getrusage(RUSAGE_SELF, &before);
        strata::kernels::PleTable table;
        strata::kernels::PleIoOptions opt; opt.mode = strata::kernels::PleIo::Mmap; opt.lock = true;
        std::string error;
        const auto start = std::chrono::steady_clock::now();
        if (!table.open(argv[1], error, opt)) throw std::runtime_error(error);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        getrusage(RUSAGE_SELF, &after);
        const auto read_after = io_bytes();
        const auto after_pages = resident((void*) begin, mapped_bytes, page);
        uint64_t checksum = 1469598103934665603ull;
        for (unsigned i = 0; i < 257; ++i) {
            const uint32_t row = (uint32_t) ((uint64_t) i * (table.rows() - 1) / 256);
            float got[160], want[160]; table.read_row(row, got);
            for (int b = 0; b < 5; ++b) strata::dequantize_q8_0(data + (size_t) row * 170 + b * 34, want + b * 32);
            if (std::memcmp(got, want, sizeof got)) throw std::runtime_error("original-byte row mismatch");
            for (size_t j = 0; j < sizeof got; ++j) checksum = (checksum ^ ((const uint8_t*) got)[j]) * 1099511628211ull;
        }
        const double cpu = after.ru_utime.tv_sec - before.ru_utime.tv_sec + after.ru_stime.tv_sec - before.ru_stime.tv_sec +
            (after.ru_utime.tv_usec - before.ru_utime.tv_usec + after.ru_stime.tv_usec - before.ru_stime.tv_usec) / 1e6;
        std::printf("{\"table_bytes\":%llu,\"pages\":%llu,\"resident_before\":%llu,\"resident_after\":%llu,"
                    "\"open_seconds\":%.6f,\"cpu_seconds\":%.6f,\"major_faults\":%ld,\"minor_faults\":%ld,"
                    "\"file_read_bytes\":%llu,\"fadvise_error\":%d,\"locked\":%s,\"row_checksum\":\"%016llx\"}\n",
                    (unsigned long long) bytes, (unsigned long long) (1 + (mapped_bytes - 1) / page),
                    (unsigned long long) before_pages, (unsigned long long) after_pages, seconds, cpu,
                    after.ru_majflt - before.ru_majflt, after.ru_minflt - before.ru_minflt,
                    (unsigned long long) (read_after - read_before), advice_error, table.locked() ? "true" : "false",
                    (unsigned long long) checksum);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
