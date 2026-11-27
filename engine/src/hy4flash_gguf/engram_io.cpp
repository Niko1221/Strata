// src/hy4flash_gguf/engram_io.cpp - the type guard between Engram's row numbers and the engine's row reader.
#include "strata/hy4flash_gguf/engram_io.hpp"

#include "strata/kernels/ngram.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace strata::hy4_flash {

EngramIo engram_io_mode() {
    static const EngramIo mode = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_ENGRAM_IO");
        if (v != nullptr && std::strcmp(v, "direct") == 0) return EngramIo::Direct;
        return EngramIo::Mmap;
    }();
    return mode;
}

struct EngramRows::Impl {
    ngram::PleReader reader;
    uint64_t n_rows = 0;
    uint32_t row_bytes = 0;
    std::vector<uint32_t> ids32;   ///< the narrowed row numbers, reused so a step allocates nothing
};

EngramRows::EngramRows() : impl_(new Impl) {}

EngramRows::~EngramRows() {
    delete impl_;
}

bool EngramRows::open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t row_bytes,
                      std::string& err) {
    if (impl_->reader.is_open()) return true;
    // Check int64 Engram row IDs before narrowing to the reader's uint32 index.
    if (n_rows == 0 || n_rows > (uint64_t) std::numeric_limits<uint32_t>::max()) {
        err = "EngramRows: " + std::to_string(n_rows) +
              " rows does not fit the reader's uint32 row numbers; staying on the mapping";
        return false;
    }
    if (row_bytes == 0 || row_bytes > ngram::PAGE) {
        err = "EngramRows: a row is " + std::to_string(row_bytes) + " bytes and the reader wants at most one page";
        return false;
    }
    // The reader's own defaults, not new ones: `PleIoOptions` is where the engine states them, and the
    // whole point of reusing it is to inherit its tuning rather than invent a second set of constants.
    const kernels::PleIoOptions io{};
    if (!impl_->reader.open(path, table_offset, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread,
                            row_bytes))
        return false;
    impl_->reader.set_keepalive(io.keepalive_ms, io.keepalive_window_s);
    impl_->n_rows = n_rows;
    impl_->row_bytes = row_bytes;
    return true;
}

bool EngramRows::is_open() const {
    return impl_->reader.is_open();
}

void EngramRows::close() {
    impl_->reader.close();
}

bool EngramRows::gather(const int64_t* ids, std::size_t n, uint8_t* out, std::string& err) {
    if (!impl_->reader.is_open()) {
        err = "EngramRows: not open";
        return false;
    }
    if (n == 0) return true;
    if (ids == nullptr || out == nullptr) {
        err = "EngramRows: null rows or destination";
        return false;
    }
    impl_->ids32.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int64_t id = ids[i];
        // Out of range becomes a row the reader will zero-fill, which is what the mapping path does for the
        // same id. `valid[]` already carries the DEAD-token meaning and is applied by `engram_layer`.
        impl_->ids32[i] = id >= 0 && (uint64_t) id < impl_->n_rows ? (uint32_t) id
                                                                   : (uint32_t) impl_->n_rows;
    }
    // S2 issues and collects together. Splitting them so the four tables' reads run under the blocks before
    // each table is S4, and it is a change to the caller's structure, not to this shell.
    const ngram::PleReader::Ticket t = impl_->reader.issue(impl_->ids32.data(), n, out);
    return impl_->reader.collect(t, err);
}

const ngram::ReaderStats* EngramRows::stats() const {
    return impl_->reader.is_open() ? &impl_->reader.stats() : nullptr;
}

}  // namespace strata::hy4_flash
