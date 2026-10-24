// include/strata/prefill/prompt_uploads.hpp - how much a prompt chunk sends up through the prompt path's pinned slots
// (header-only: it is arithmetic).
//
// The prompt path's pageable uploads go up from a ring of pinned slots (#1996, core::PinnedRing), and a slot grows at
// its first larger upload.  A read beside the batch slots (STRATA_PIPED_CONCURRENT) grows them before the read instead,
// since an allocation or a free on the read's thread takes the driver's lock (#31): to the largest upload one chunk
// can send.  The picture rows' upload and that reservation both take their sizes from here.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

/// A picture's consecutive rows are gathered into one pinned slot and go up as one copy, at most this many bytes.
constexpr int64_t kPictureRunBytes = int64_t(4) << 20;

/// Rows of `row_bytes` bytes that one picture upload takes (at least one).
inline int64_t picture_run_rows(int64_t row_bytes) {
    return std::max<int64_t>(1, kPictureRunBytes / std::max<int64_t>(row_bytes, 1));
}

/// The largest upload a chunk of `T` tokens sends through the pinned slots: the routing tables (T x `k` ids), the PLE
/// rows when they could not be pinned (T x `row_floats`), and with `pictures` a run of a picture's rows (`row_floats`
/// each; a chunk holds at most T of them).
inline int64_t prompt_upload_bytes(int64_t T, int64_t k, int64_t row_floats, bool ple_pageable, bool pictures) {
    const int64_t row_bytes = row_floats * (int64_t) sizeof(float);
    int64_t bytes = T * k * (int64_t) sizeof(int32_t);
    if (ple_pageable) bytes = std::max(bytes, T * row_bytes);
    if (pictures) bytes = std::max(bytes, std::min(T, picture_run_rows(row_bytes)) * row_bytes);
    return bytes;
}

}  // namespace strata::prefill
