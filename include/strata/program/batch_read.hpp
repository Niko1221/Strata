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

}  // namespace strata::program
