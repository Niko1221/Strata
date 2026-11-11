#pragma once
// # Standalone PLE metrics module.
// Keeps request-level snapshot/delta/format logic out of the upstream PLE and generate code.
// Upstream injection points only call io_snapshot() and request_line().

#include <cstdint>
#include <cstdio>
#include <string>

#include "strata/ngram/ple_reader.hpp"

namespace strata::core::ple_metrics {

/// Cumulative, consistent PLE reader counters. Take one before and after a request; `delta()` attributes the work.
struct PleIoSnapshot {
    bool direct = false;
    bool locked = false;
    uint64_t requests = 0;       ///< rows asked for (16 a token, before dedupe and cache)
    uint64_t cache_hits = 0;     ///< rows served by the row cache
    uint64_t dedup_rows = 0;     ///< rows that shared a page already read in the same ticket
    uint64_t reads = 0;          ///< SSD read requests submitted (keep-alives excluded)
    uint64_t bytes = 0;          ///< bytes actually returned by those reads
    uint64_t keepalive_reads = 0;
    double wait_us = 0;          ///< time blocked in collect()
    double submit_us = 0;        ///< time spent submitting reads
    double read_us_sum = 0;      ///< sum of per-read latencies (keep-alives excluded)
    uint64_t cache_size = 0;
    uint64_t cache_capacity = 0;
};

inline PleIoSnapshot from_reader(const strata::ngram::ReaderStats& s, bool direct, bool locked,
                                 uint64_t cache_size, uint64_t cache_capacity) {
    PleIoSnapshot out;
    out.direct = direct;
    out.locked = locked;
    out.requests = s.requests;
    out.cache_hits = s.cache_hits;
    out.dedup_rows = s.dedup_rows;
    out.reads = s.reads;
    out.bytes = s.bytes;
    out.keepalive_reads = s.keepalive_reads;
    out.wait_us = s.wait_us;
    out.submit_us = s.submit_us;
    out.read_us_sum = s.read_us_sum;
    out.cache_size = cache_size;
    out.cache_capacity = cache_capacity;
    return out;
}

inline PleIoSnapshot delta(const PleIoSnapshot& before, const PleIoSnapshot& after) {
    PleIoSnapshot d;
    d.direct = after.direct;
    d.locked = after.locked;
    const auto u = [](uint64_t a, uint64_t b) { return b >= a ? b - a : 0; };
    const auto f = [](double a, double b) { return b >= a ? b - a : 0.0; };
    d.requests = u(before.requests, after.requests);
    d.cache_hits = u(before.cache_hits, after.cache_hits);
    d.dedup_rows = u(before.dedup_rows, after.dedup_rows);
    d.reads = u(before.reads, after.reads);
    d.bytes = u(before.bytes, after.bytes);
    d.keepalive_reads = u(before.keepalive_reads, after.keepalive_reads);
    d.wait_us = f(before.wait_us, after.wait_us);
    d.submit_us = f(before.submit_us, after.submit_us);
    d.read_us_sum = f(before.read_us_sum, after.read_us_sum);
    d.cache_size = after.cache_size;
    d.cache_capacity = after.cache_capacity;
    return d;
}

inline std::string request_line(const PleIoSnapshot& before, const PleIoSnapshot& after) {
    const PleIoSnapshot d = delta(before, after);
    const char* mode = d.locked ? "ram" : (d.direct ? "direct" : "mmap");
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "strata serve: PLE this request: mode=%s rows=%llu cache_hits=%llu (%.1f%%) dedup=%llu "
                  "disk_reads=%llu disk_read=%.2f MB avg_read=%.0f us blocked=%.1f ms submit=%.1f ms cache=%llu/%llu",
                  mode, (unsigned long long) d.requests, (unsigned long long) d.cache_hits,
                  d.requests ? 100.0 * (double) d.cache_hits / (double) d.requests : 0.0,
                  (unsigned long long) d.dedup_rows, (unsigned long long) d.reads, (double) d.bytes / 1e6,
                  d.reads ? d.read_us_sum / (double) d.reads : 0.0, d.wait_us / 1000.0, d.submit_us / 1000.0,
                  (unsigned long long) d.cache_size, (unsigned long long) d.cache_capacity);
    return buf;
}

/// Template so this module needs no complete PleTable type; instantiated where the caller includes ngram.hpp.
template <typename Table>
inline std::string request_line(const Table* table, const PleIoSnapshot& before) {
    return table != nullptr ? request_line(before, table->io_snapshot()) : std::string{};
}

}  // namespace strata::core::ple_metrics
