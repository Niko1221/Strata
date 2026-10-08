#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

// How many prompt tokens one run of a --batch prompt read covers.  Beside decoding slots (and on one GPU) it is one
// prompt chunk, so the slots decode between chunks.  A layer split with no slot decoding reads the rest as ONE run,
// which pipelines its stages over the chunks (returned as 0) -- unless `piece` (STRATA_SPLIT_PIECE) is set: then it
// reads that many tokens (a whole number of chunks) per run, so a waiting shorter request's BYIELD is seen within one
// piece instead of after the whole read.
inline int64_t prompt_read_piece(int64_t chunk, bool split_idle, int64_t piece) {
    chunk = std::max<int64_t>(chunk, 1);
    if (!split_idle) return chunk;
    if (piece <= 0) return 0;
    return std::max(chunk, piece / chunk * chunk);
}

// Where one run of that read ends: `piece` from prompt_read_piece, decided again at every boundary, so a read that
// started beside a decoding slot goes back to whole pieces (or the whole rest) once no slot decodes.
inline int64_t prompt_read_run_end(int64_t q, int64_t end, int64_t piece) {
    return piece <= 0 ? end : std::min(end, q + piece);
}

} // namespace strata::program
