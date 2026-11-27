// engine/metal/src/hy4flash_gguf/model.h - hy4-flash's weights for the Metal engine.
//
// THE TWO GGUF READERS HAVE DIFFERENT JOBS, and mixing them up crashes the process:
//   * `strata::GgufModel` (include/strata/artifact/gguf_reader.hpp, header-only) reads the METADATA of BOTH
//     shards.  It is the only one that may touch shard 2, whose single tensor is the 59 GiB Engram table in
//     F8_E4M3 (ggml type 43).
//   * `gguf::File` (metal/src/gguf_lite.h) gives the TENSOR OFFSETS, and only for SHARD 1: its
//     `ref::block_bytes` calls `std::exit(2)` on a type it does not know, and 43 is one of those.
//
// WHAT IS RESIDENT AND WHAT IS NOT.  `load_dense` reads everything but the routed experts into one anonymous
// Metal buffer with F_NOCACHE preads (no file cache, so the pages are the engine's and nothing else's), and
// rewrites each tensor's `offset` to its offset inside that buffer.  Left out:
//   * `*_exps` - the 76.5 GB of routed experts, streamed from the SSD per layer (stream.h);
//   * `per_layer_token_embd.weight` - the Engram table, read row by row from shard 2 (ngram/ple_reader.hpp);
//   * every tensor of the MTP block when it is not in use - it is 6.75 GB of BF16 experts plus 0.1 GB of dense
//     weights.  `with_mtp` turns it back on, which is why the skip list is built from the geometry's own
//     `trunk_blocks` rather than from the literal name "blk.42".
//
// So a tensor's `offset` means one of two things depending on when it is read, and the only safe order is:
// build the expert table first (its tensors are `*_exps`, which load_dense never rewrites), then load.
#pragma once

#include "engine/stream.h"
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "strata/hy4flash_gguf/geometry.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::hy4_flash {

enum { kGate = 0, kUp = 1, kDown = 2 };

/// One expert ordinal's three weight slices, as byte ranges of the GGUF.
///
/// THE ORDINAL IS NOT THE BLOCK ID (include/strata/hy4flash_gguf/geometry.hpp): block 0 is the dense FFN and has no
/// experts, blocks 1..41 are ordinals 0..40, and the MTP block takes the last ordinal, 41.  Every per-expert
/// statistic, the profile and the arena key off the ordinal, so an off-by-one here reads another block's experts.
struct ExpertSlices {
    int32_t ordinal = -1;
    int32_t block_id = -1;
    uint64_t off[3] = {0, 0, 0};      ///< gate / up / down: expert 0's byte offset from the start of the file
    uint64_t stride[3] = {0, 0, 0};   ///< one expert's bytes (expert e starts at off + e * stride)
    int32_t type[3] = {-1, -1, -1};   ///< the ggml type, which differs per block: 22 blocks store gate/up as
                                      ///< Q3_K and 19 as Q4_K, down is Q4_K throughout, the MTP block BF16
    int32_t n_experts = 0;
};

struct Model {
    hy4_flash::Geometry geo;
    std::string shard1_path;
    std::string engram_path;             ///< the shard that holds per_layer_token_embd.weight (shard 2 here)
    uint64_t engram_offset = 0;          ///< row 0's byte offset in `engram_path` (data_start + the tensor's)
    uint64_t engram_rows = 0;            ///< the merged table's row count
    uint32_t engram_row_bytes = 0;       ///< one row in the file (102 for Q8_0 x 96; F8_E4M3 differs)
    gguf::File tensors;                  ///< SHARD 1 only (see the header comment)
    std::vector<ExpertSlices> experts;   ///< geo.n_expert_ordinals() entries, indexed by ordinal
    std::vector<std::string> skipped;    ///< the MTP block's tensors, by name (and the on_file ones below)
    /// Tensors kept OUT of the dense buffer and read from shard 1 when needed (absolute file offsets), so their bytes
    /// go to the expert arena instead: token_embd (one row a token is all a window reads) and, with the MTP
    /// block quantized at load, its BF16 matrices (read once to build the Q8_0 copies).  HY4_FLASH_EMBD_RESIDENT=1 keeps
    /// token_embd in the dense buffer instead.
    std::map<std::string, uint64_t> on_file;
    /// `len` bytes of tensor `name` from byte `at` of it: the dense buffer when resident, else a pread of shard 1.
    bool tensor_bytes(const std::string& name, uint64_t at, uint64_t len, void* dst) const;
    uint64_t dense_plan_bytes = 0;       ///< what the dense tensors will need, from the header alone
    int32_t dense_plan_tensors = 0;
    uint64_t dense_bytes = 0;            ///< what was read (0 until load_dense)
    double load_s = 0;
    mtl::Buffer dense;
    strata::metal_engine::ExpertStream stream;

    /// Shard 1's entry for `name`, or nullptr.  After `load_dense` a dense tensor's `offset` is its offset in
    /// `dense`, not in the file.
    const gguf::Tensor* find(const std::string& name) const { return tensors.find(name); }
};

/// Reads the geometry of the model whose shard 1 is `shard1`, and plans the dense residency: the skip list and
/// the expert table.  Reads no payload.  Returns "" or the first problem, named.
///
/// `with_mtp` keeps the MTP block's DENSE tensors resident (238 MB of BF16 here); its routed experts
/// stay out either way - they are `*_exps` and the MTP module reads them itself (mtp.h).
std::string open(Model& m, const std::string& shard1, int32_t max_context, bool with_mtp = false);

/// Reads the dense tensors into one resident buffer (F_NOCACHE preads on 4 threads) and rewrites their offsets.
/// Must run after `open`.  Returns "" or the problem.
std::string load_dense(Model& m, mtl::Context& cx);

/// One line per fact, for `--help`-style output: the geometry, the dense plan and the expert types per ordinal.
std::string describe(const Model& m);

}  // namespace strata::hy4_flash
