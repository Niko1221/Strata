// engine/metal/src/hy4flash_gguf/dump.h - the node dump the block-parity check reads.
//
// The format is the reference's own (tmp/tests/hy4_flash/dump/dump_ref.cpp, "dump_reference v3"): one `.bin` per node,
// contiguous in its declared shape, and a TAB-separated `manifest.txt` whose header carries the token ids.
// tmp/tests/hy4_flash/pyref/check_block.py reads both dumps through one class, so the engine's node NAMES and SHAPES
// have to be the reference's exactly - `ffn_moe_weights_scaled` is [1, topk, T, 1] and not [topk, T, 1, 1], and
// getting that wrong makes the check report a shape mismatch rather than a number.
//
// A node arrives one window at a time (the engine processes at most 8 positions at once), so the per-position
// nodes are accumulated in host memory and written when the run ends.  A 4608-position dump of a block's nodes is
// about 0.7 GB, which is why `close()` writes and frees rather than keeping everything.
//
// SEVERAL RUNS WRITE ONE DUMP.  tmp/tests/hy4_flash/engine/block_test.sh runs one engine invocation per block into the
// same directory, and the check needs all of them in one manifest - so `open()` reads an existing manifest and
// `close()` carries over every row this run did not produce.  A node's file is named after the node (not by an
// index), so two runs cannot collide on a file name while disagreeing about which node is in it.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::hy4_flash {

class Dump {
public:
    /// `dir` is created if it does not exist.  `tokens` goes into the manifest header.
    std::string open(const std::string& dir, const std::vector<int32_t>& tokens);
    bool on() const { return !dir_.empty(); }

    /// Appends `n_pos` positions of a per-position node.  `ne` is the shape of ONE position in ggml order with
    /// the position axis left out, and `pos_axis` says where the position axis goes in the full shape (the
    /// reference's own choice per node, which check_block.py's POS_AXIS mirrors): 1 for a [..., T] node whose
    /// leading dims are its own, 2 for a plain [n, T] row.
    void add(const std::string& name, const char* type, std::vector<int64_t> ne_one, int pos_axis, const void* data,
             size_t bytes_per_pos, int n_pos);

    /// A node that is NOT per position (the indexer's pooled keys, which are per pool group).
    void add_whole(const std::string& name, const char* type, std::vector<int64_t> ne, const void* data,
                   size_t bytes);

    /// Writes every `.bin` and the manifest.  Returns "" or the problem.
    std::string close();

private:
    struct Node {
        std::string type;
        std::vector<int64_t> ne_one;
        int pos_axis = 2;
        int n_pos = 0;
        bool whole = false;
        std::vector<uint8_t> data;
    };
    struct Kept {
        std::string line;   ///< a manifest row from an earlier run, carried over verbatim
        std::string name;
    };
    std::string dir_;
    std::vector<int32_t> tokens_;
    std::vector<std::string> order_;
    std::map<std::string, Node> nodes_;
    std::vector<Kept> kept_;
};

}  // namespace strata::hy4_flash
