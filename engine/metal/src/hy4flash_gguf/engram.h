// engine/metal/src/hy4flash_gguf/engram.h - the Engram tables' rows for a window, read while the GPU is busy.
//
// WHERE THE WAIT POINT GOES, and why it is a separate thing from the expert cut.  The Engram injection happens
// BEFORE a block's attention pre-mix (src/hy4flash_gguf/forward.cu, engram.hpp's warning), while the expert stream's
// cut sits AFTER that block's router.  So joining the Engram reads at the cut of ordinal 0 / 10 / 20 / 30 would
// be TOO LATE: the dispatch that consumes table t's rows has already been encoded.  Hence this class:
//
//   issue(window)         the four tables' row ids are hashed and four readers start, one per table;
//   join(block_id)        waits for THAT block's table and hands back its rows.  It must be called before the
//                         block's first Engram dispatch is encoded, and `rows()` refuses to answer before it.
//
// There are four tables, at blocks 1 / 11 / 21 / 31, and a window asks each of them for n_tok x 16 rows of about a
// hundred bytes - a few dozen page reads.  `wait_us()` reports what each join cost, which is what tells you
// whether joining all four before block 0 would be better than joining each one just before its block.
//
// THE READS ARE O_DIRECT (`EngramIo::Direct`), FORCED.  The default for this path is the GGUF mapping, whose page
// cache is faster on a warm cache - but the table is 59 GiB and the arena rules keep no room for it in the file
// cache, so mapping it would push the arena's own pages out.  Nothing else in this engine reads through the page
// cache either (dense and the experts are F_NOCACHE preads).
#pragma once

#include "model.h"
#include "strata/hy4flash_gguf/engram.hpp"
#include "strata/hy4flash_gguf/engram_io.hpp"

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace strata::hy4_flash {

class EngramSupply {
public:
    EngramSupply() = default;
    ~EngramSupply();
    EngramSupply(const EngramSupply&) = delete;
    EngramSupply& operator=(const EngramSupply&) = delete;

    /// Opens one reader per table on the Engram shard and loads the token map from the pack.  Returns "" or the
    /// problem.  `pack_dir` is a run-time dependency of the forward pass: the map has 120,832 entries, too many
    /// for the shared metadata reader's array cap.
    std::string open(const Model& m, const std::string& pack_dir);
    bool ready() const { return !tables_.empty(); }
    int n_tables() const { return (int) tables_.size(); }
    int block_of(int table) const;

    /// Hashes the window's row ids and starts the four reads.  `tokens[i]` is the token at `pos[i]`; `prev` holds
    /// `geo.engram_predecessors()` predecessors per token, oldest first (engram.hpp's indexing).  Any previous
    /// window's reads are joined first, so a caller cannot overlap two windows by accident.
    std::string issue(const int32_t* tokens, const int32_t* pos, const int32_t* prev, int n_tok);

    /// Waits for the table at `block_id`, then returns its rows: `n_tok * columns` rows of `row_bytes`, token
    /// major.  nullptr when that block has no table or the read failed (`err` says which).
    const uint8_t* join(int block_id, std::string& err);
    /// The validity mask the hash produced: `n_tok * columns` floats, 0 for a dead token's column (engram.hpp,
    /// whose `engram_row_ids` hands it back as floats because that is what the injection multiplies by).
    const float* valid(int block_id) const;
    uint64_t rows_per_token() const { return (uint64_t) columns_; }
    uint32_t row_bytes() const { return row_bytes_; }
    /// How long `join` waited for that block, in microseconds.
    double wait_us(int block_id) const;
    void join_all();

    /// WHERE A WINDOW'S ENGRAM TIME GOES.  The engine's `engram_wait_ms` covers everything from before
    /// `prepare_window` to the last join; this splits it into three pieces, because they have three different
    /// causes:
    ///
    ///   tail_us   `issue` begins with `join_all()`, so a reader of the PREVIOUS window that has not finished is
    ///             charged here.  A zero here is what licenses reading `join_us` as this window's own wait.
    ///   hash_us   `engram_row_ids` for the four tables plus the four std::thread creations: pure CPU, and a
    ///             function of the window size only, so it must NOT grow with the context.
    ///   join_us   waiting for THIS window's reads, i.e. the SSD queue the expert stream shares.
    ///
    /// Accumulated until `reset_timing()`.  A prompt's waits belong in `Timing::pf_engram_ms` and must not reach
    /// these counters: the print divides them by the DECODE position count, so a prompt's time charged here
    /// inflates the per-position figure instead of appearing as prefill.
    struct Timing {
        double tail_us = 0, hash_us = 0, join_us = 0;
        uint64_t windows = 0, joins = 0;
    };
    const Timing& timing() const { return tm_; }
    void reset_timing() { tm_ = Timing(); }

    /// WHAT A WINDOW'S WAIT ACTUALLY DEPENDS ON, which `Timing` alone cannot say.  It is not the table's size,
    /// which does not grow with the context, and not the SSD queue the expert stream shares: it is WHICH ROWS
    /// are asked for.  `PleReader` has a row cache and coalesces rows that share a page, so a context whose
    /// tokens are repetitive generates n-grams that hash to rows it already holds.  These counters say that
    /// directly instead of by inference - summed over the tables, because a window waits for all four.
    struct IoStats {
        uint64_t requests = 0, cache_hits = 0, dedup_rows = 0, reads = 0, bytes = 0;
        double wait_us = 0, read_us_sum = 0;   ///< `wait_us` is blocked time, `read_us_sum` the latencies
        double p50_us = 0, p99_us = 0;         ///< of the per-read latencies the reader keeps
        /// Rows served without an SSD read.  A tok/s budget that divides bytes by the bandwidth is wrong when
        /// this is high, exactly as it is for the expert arena.
        double hit_rate() const { return requests ? (double) cache_hits / (double) requests : 0.0; }
        /// Latency added per read on average.  `join_us / reads` would mix in the thread handshake; this is
        /// the reader's own number.
        double us_per_read() const { return reads ? read_us_sum / (double) reads : 0.0; }
    };
    IoStats io_stats() const;

private:
    struct Table;
    Table* find(int block_id) const;
    // Raw pointers, not unique_ptr: `Table` is only complete inside engram.cpp, and a unique_ptr member
    // makes every translation unit that merely holds an EngramSupply instantiate its deleter.
    std::vector<Table*> tables_;
    std::vector<int32_t> token_map_;
    Timing tm_;
    int columns_ = 0, n_tok_ = 0;
    uint32_t row_bytes_ = 0;
};

}  // namespace strata::hy4_flash
