// Read Engram rows through the engine's shared row reader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::ngram {
struct ReaderStats;
}

namespace strata::hy4_flash {

/// How a step's Engram rows reach the host.
enum class EngramIo {
    Mmap,     ///< through the GGUF mapping, so the page cache serves repeats. The engine's existing path.
    Direct,   ///< `strata::ngram::PleReader`: O_DIRECT, one persistent I/O worker, bounded row cache,
              ///< page deduplication, `max_inflight` queue depth and per-read latency percentiles.
};

/// STRATA_HY4_FLASH_ENGRAM_IO selects mmap (default) or direct I/O.
EngramIo engram_io_mode();

/// Keep the file and reader workers alive across forward passes.
class EngramRows {
public:
    EngramRows();
    ~EngramRows();
    EngramRows(const EngramRows&) = delete;
    EngramRows& operator=(const EngramRows&) = delete;

    /// `table_offset` is row 0's byte offset in `path`, `n_rows` the table's row count and `row_bytes` one
    /// row (102 for this model's Q8_0 x 96). All three come from the GGUF directory, never hard-coded.
    /// False with `err` leaves the caller on its mapping path, which is not an error.
    bool open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t row_bytes,
              std::string& err);
    bool is_open() const;
    void close();

    /// `n` rows' raw bytes into `out`, which must hold `n * row_bytes`. `ids` are the global row numbers
    /// `engram_row_ids` produced. An id outside `[0, n_rows)` yields zero bytes - the mapping's own
    /// behaviour, and distinct from `engram.hpp`'s `valid[]` mask, which is about DEAD tokens.
    bool gather(const int64_t* ids, std::size_t n, uint8_t* out, std::string& err);

    /// The reader's own counters: requests, cache_hits, dedup_rows, reads, bytes, wait_us, percentiles.
    /// Null when closed. Reported as the reader gives them; this path invents no metrics of its own.
    const ngram::ReaderStats* stats() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace strata::hy4_flash
