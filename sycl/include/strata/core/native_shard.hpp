// include/strata/core/native_shard.hpp - which bytes of a GGUF matrix THIS RANK keeps, as pure arithmetic.
//
// THE NATIVE FORM IS THE GGUF BLOCK LAYOUT, so a matrix's payload is a whole number of blocks with `shape[0]`
// (ne0, the contiguous axis) varying fastest: one row is `full_bytes / ne1` bytes and a run of rows is a
// contiguous run of bytes.  That is the whole reason this is a few lines and not a gather:
//
//   * an OUTPUT split keeps whole rows - one run per range;
//   * an INPUT (contraction) split keeps the same block range inside EVERY row - one run per row per range.
//     A row's kept part starts at `row * row_bytes + (k0 / block_elements) * block_bytes`, which is exactly
//     where the engine form puts it when the runs are packed in this order, so the destination needs no
//     offsets of its own.
//
// It lives in a header, and takes plain numbers rather than a GGUF reader, so the boundaries and the refusals
// can be tested on a machine with no GGUF file, no model and no GPU - which is the point: the arithmetic here
// is what decides whether a shard is the bytes it claims.
#pragma once

#include "strata/core/shard.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One contiguous run of a matrix's SOURCE bytes.  The destination is the runs packed in this order.
struct ShardRun {
    uint64_t src = 0;
    uint64_t len = 0;
};

/// `ne0`/`ne1` are the FULL matrix's extents (the GGUF tensor's own shape), `full_bytes` its whole payload in
/// the native form, and `block_elements`/`block_bytes` come from its type (`strata::block_geometry`).  `ts`
/// is null (or replicated) on a rank that keeps the whole matrix.
///
/// Every refusal is a split this code cannot represent correctly, and each one would otherwise produce a
/// matrix of the right SHAPE built from the wrong bytes:
///   * a zero extent, or a payload that is not a whole number of rows;
///   * a range outside the axis it splits;
///   * a contraction range that is not a whole number of the FORMAT's blocks - note the block size, not the
///     128-element activation block `shard_check_pair` checks: a K-quant super-block is 256, and a split
///     inside one would decode to a plausible matrix rather than fail.
inline bool shard_runs(uint64_t ne0, uint64_t ne1, uint64_t full_bytes, int block_elements, int block_bytes,
                       const TensorShard* ts, std::vector<ShardRun>& out, std::string& err) {
    out.clear();
    if (ne0 == 0 || ne1 == 0 || full_bytes == 0) {
        err = "a matrix with no extent";
        return false;
    }
    if (full_bytes % ne1 != 0) {
        err = "a payload of " + std::to_string(full_bytes) + " B that is not a whole number of " +
              std::to_string(ne1) + " rows";
        return false;
    }
    const uint64_t row_bytes = full_bytes / ne1;
    if (ts == nullptr || ts->axis == ShardAxis::Replicated || ts->axis == ShardAxis::Rank0) {
        out.push_back(ShardRun{0, full_bytes});
        return true;
    }
    if (ts->ranges < 1) {
        err = "a split with no ranges";
        return false;
    }
    if (ts->axis == ShardAxis::Output) {
        for (int i = 0; i < ts->ranges; ++i) {
            const ShardRange& r = ts->range[i];
            if (r.count() <= 0 || r.begin < 0 || r.end > (int64_t) ne1) {
                err = "a kept row range outside the matrix's " + std::to_string(ne1) + " rows";
                return false;
            }
            out.push_back(ShardRun{(uint64_t) r.begin * row_bytes, (uint64_t) r.count() * row_bytes});
        }
        return true;
    }
    if (block_elements <= 0 || block_bytes <= 0) {
        err = "no block geometry (the format's block size) for a contraction split";
        return false;
    }
    for (uint64_t row = 0; row < ne1; ++row) {
        for (int i = 0; i < ts->ranges; ++i) {
            const ShardRange& r = ts->range[i];
            if (r.count() <= 0 || r.begin < 0 || r.end > (int64_t) ne0 ||
                r.begin % block_elements != 0 || r.count() % block_elements != 0) {
                err = "a kept contraction range that is not a whole number of " + std::to_string(block_elements) +
                      "-element blocks inside the matrix's " + std::to_string(ne0) + " columns";
                return false;
            }
            out.push_back(ShardRun{row * row_bytes +
                                       (uint64_t) (r.begin / block_elements) * (uint64_t) block_bytes,
                                   (uint64_t) (r.count() / block_elements) * (uint64_t) block_bytes});
        }
    }
    return true;
}

/// The engine form's size of those runs: the destination size, and the check that the copy is complete.
inline uint64_t shard_runs_bytes(const std::vector<ShardRun>& runs) {
    uint64_t n = 0;
    for (const ShardRun& r : runs) n += r.len;
    return n;
}

}  // namespace strata::core
