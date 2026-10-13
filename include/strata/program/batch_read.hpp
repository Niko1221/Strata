// include/strata/program/batch_read.hpp - how --batch reads a prompt in pieces (header-only: it is arithmetic).
//
// A prompt read with --batch stops between two pieces to (a) let the slots decode and (b) look for a BYIELD (a
// shorter request is waiting).  The pieces always end on a multiple of the prompt chunk, so they read exactly the
// chunks a single run would.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

/// Tokens of the next piece of a --batch prompt read, or 0 for "the whole rest in one run".
///   beside_slots: some slot is decoding (its windows run between the pieces)
///   layer_split / piped: the read goes through a layer split's stages / the pipelined groups
///   chunks_per_piece: for a layer split with no slot decoding, the pieces' length in chunks (0: one run)
/// A layer split reads its stages as a pipeline over one run's chunks, so a piece costs a pipeline fill; with slots
/// decoding the pieces are single chunks anyway.  With no slot decoding a piece is `chunks_per_piece` chunks: long
/// enough to keep the pipeline full, short enough that a BYIELD is taken within a few chunks (it used to wait for the
/// whole prompt, and the long prompt then held its slot: "ERR BGEN: no such free slot" for the request that should
/// have taken over).
inline int64_t batch_read_piece(bool batch, bool piped, bool layer_split, bool beside_slots, int64_t chunk,
                                int64_t chunks_per_piece) {
    if (!batch || piped) return 0;
    chunk = std::max<int64_t>(chunk, 1);
    if (!layer_split || beside_slots) return chunk;
    return chunks_per_piece > 0 ? chunk * chunks_per_piece : 0;
}

/// What a configuration has that keeps the concurrent read (STRATA_PIPED_CONCURRENT=1) off.
struct ConcurrentReadSetup {
    bool prefill_help = false;      ///< STRATA_PREFILL_HELP: the idle stage's help allocates on the read's thread
    bool split_mtp_batch = false;   ///< STRATA_SPLIT_MTP_BATCH: the batched draft K/V frees there (a free waits)
    bool fs_slots = false;          ///< STRATA_FS_SLOTS (Foresight swap space): serving a doorbell makes a CUDA call
    bool pcie_dma = false;          ///< --pcie-mode dma: the same (its copies)
    bool helper_caches = false;     ///< --expert-cache-device1..3 (and --remote-expert-opt on them), a prompt source of
                                    ///< its own; --remote-expert-opt alone is inert on a plain layer split
    bool staged_source = false;     ///< the GGUF source stages its blobs (in place, unbuffered, STRATA_IO_PREFETCH):
                                    ///< it recycles the buffers it hands out, one thread at a time
    bool other_source = false;      ///< an expert source that is neither the GGUF source nor the arena
};

/// Why the concurrent read stays off with `s` (the first reason, for the log line), or nullptr when it can run.
inline const char* concurrent_read_off(const ConcurrentReadSetup& s) {
    if (s.prefill_help) return "STRATA_PREFILL_HELP";
    if (s.split_mtp_batch) return "STRATA_SPLIT_MTP_BATCH";
    if (s.fs_slots) return "STRATA_FS_SLOTS";
    if (s.pcie_dma) return "--pcie-mode dma";
    if (s.helper_caches) return "the helper caches (--expert-cache-device1..3)";
    if (s.staged_source)
        return "an expert source that stages its blobs (the GGUF in place, unbuffered, STRATA_IO_PREFETCH)";
    if (s.other_source) return "this expert source";
    return nullptr;
}

}  // namespace strata::program
