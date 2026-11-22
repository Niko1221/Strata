// include/strata/core/mtp_head_rule.hpp - the bytes of one row of the MTP draft head's subset buffer (src/core/mtp.cpp bind / make_q4_head).
//
// #5841: --mtp-q4 head|all rewrites the gathered subset in place as Q4_0 rows (n_embd / 32 * 18 bytes).  The buffer was sized for the head's own row, and a
// head in IQ4_XS (n_embd / 256 * 136 bytes: 1,360 against Q4_0's 1,440 at n_embd 2560) has the narrower row, so the host-to-device copy wrote 8.1 MiB past
// the buffer ("mtp: writing the draft head", exit 1).  The buffer takes the wider of the two rows.  Pure: no GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::core {

/// the bytes a subset row needs: the head's own row, or the Q4_0 row when --mtp-q4 rewrites the subset (and n_embd is a multiple of 32), whichever is wider
inline int64_t mtp_draft_head_row_cap(int64_t head_row_bytes, int64_t n_embd, bool q4_head) {
    const int64_t q4_row = (q4_head && n_embd > 0 && n_embd % 32 == 0) ? (n_embd / 32) * 18 : 0;
    return std::max(head_row_bytes, q4_row);
}

/// STRATA_MTP_HEAD_INDEX=1: whether bind reads the subset's rows in the main head by index (no copy).  MtpDrafter::bind and
/// MtpDrafter::bind_bytes both ask this one rule, so the cache is sized for what bind allocates: where it says no (a head
/// type without native_mmvq_rows - every HIP build -, --mtp-q4 head|all, a head on another card or moved to the host, an id
/// outside the vocabulary) bind copies the rows and bind_bytes counts them.  `ids` = draft_vocab.bin's bytes (int32 ids).
inline bool mtp_draft_head_by_index(bool env, bool q4_head, bool head_on_host, int head_device, int drafter_device,
                                    bool type_supported, const uint8_t* ids, size_t id_bytes, int64_t n_vocab) {
    if (!env || q4_head || head_on_host || (head_device >= 0 && head_device != drafter_device) || !type_supported ||
        ids == nullptr || id_bytes < 4 || id_bytes % 4 != 0)
        return false;
    for (size_t i = 0; i < id_bytes; i += 4) {
        const int32_t id = (int32_t) ((uint32_t) ids[i] | (uint32_t) ids[i + 1] << 8 | (uint32_t) ids[i + 2] << 16 |
                                      (uint32_t) ids[i + 3] << 24);
        if (id < 0 || id >= n_vocab) return false;
    }
    return true;
}

/// The VRAM of the draft head's rows: what bind allocates (row = its buffer's row, mtp_draft_head_row_cap) and what
/// bind_bytes prices (row = the head's row) - none when mtp_draft_head_by_index says the rows are read in the head.
inline uint64_t mtp_draft_head_rows_bytes(uint64_t n_ids, uint64_t row_bytes, bool by_index) {
    return by_index ? 0 : n_ids * row_bytes;
}

}  // namespace strata::core
