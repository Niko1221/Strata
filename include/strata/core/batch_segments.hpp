#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

inline constexpr int kBatchMaxRows = 16;

// Give every active request its pending row, then rotate access to spare rows.
// Callers bound each wanted width to four, preserving two <=8-row groups.
inline void allocate_batch_rows(const int* wanted, int count, int budget, int first, int* out) {
    for (int i = 0; i < count; ++i) {
        out[i] = wanted[i] > 0 ? 1 : 0;
        budget -= out[i];
    }
    while (budget > 0) {
        bool changed = false;
        for (int j = 0; j < count && budget > 0; ++j) {
            const int i = (first + j) % count;
            if (out[i] < wanted[i]) { ++out[i]; --budget; changed = true; }
        }
        if (!changed) break;
    }
}

// Preserve complete causal segments while bounding each execution group to the
// existing eight-row kernels. Zero means no second group, -1 means unsupported.
inline int segment_group_boundary(const int* slots, int count) {
    if (count <= 8) return 0;
    for (int i = 8; i > 0; --i)
        if (slots[i - 1] != slots[i] && count - i <= 8) return i;
    return -1;
}

// ngram_rows consumes a separate oldest-first pair per row. Its API does not
// advance a single predecessor pair across a sequence.
inline void segment_predecessors(const int32_t* tokens, int width, const int32_t* initial, int32_t* pairs) {
    int32_t older = initial[0], newer = initial[1];
    for (int i = 0; i < width; ++i) {
        pairs[2 * i] = older;
        pairs[2 * i + 1] = newer;
        older = newer;
        newer = tokens[i];
    }
}

inline int accepted_segment_prefix(const int32_t* inputs, const int32_t* outputs, int width,
                                    const int64_t* eos_ids, int eos_count) {
    if (!inputs || !outputs || width < 1) return 0;
    int accepted = 1;
    while (accepted < width && inputs[accepted] == outputs[accepted - 1]) ++accepted;
    for (int i = 0; i < accepted; ++i)
        for (int j = 0; j < eos_count; ++j)
            if (outputs[i] == eos_ids[j]) return i + 1;
    return accepted;
}

// Packed rows of one slot form one contiguous causal segment. Independent
// slots may appear in any order, but a slot may not reappear after its segment.
inline int segment_width(const int* slots, int count, int first) {
    int end = first + 1;
    while (end < count && slots[end] == slots[first]) ++end;
    return end - first;
}

inline bool validate_segments(const int* slots, const int64_t* positions, int count,
                              int capacity, int slot_count, bool allow_multiple,
                              std::string& err) {
    if (!slots || !positions || count < 1 || count > capacity || slot_count < 1) {
        err = "verify: invalid segment buffers or row count";
        return false;
    }
    for (int first = 0; first < count;) {
        const int width = segment_width(slots, count, first);
        if (slots[first] < 0 || slots[first] >= slot_count || (!allow_multiple && width != 1)) {
            err = "verify: invalid slot or multiple rows in ordinary batching";
            return false;
        }
        for (int i = 0; i < first; ++i) if (slots[i] == slots[first]) {
            err = "verify: slot segments must be contiguous";
            return false;
        }
        for (int i = first; i < first + width; ++i) {
            // Positions are uploaded as int32. Compare without signed overflow.
            if (positions[i] < 0 || positions[i] > INT32_MAX ||
                (i > first && positions[i] - positions[i - 1] != 1)) {
                err = "verify: invalid or nonconsecutive segment positions";
                return false;
            }
        }
        first += width;
    }
    return true;
}

// keep[first] is the accepted input prefix; continuation rows must hold zero.
// At least the pending input token is committed. A rejected draft is never
// counted merely because its target output was computed.
inline bool validate_segment_keeps(const int* slots, int count, const int* keep,
                                   std::string& err) {
    if (!slots || !keep || count < 1) { err = "verify: missing segment commits"; return false; }
    for (int first = 0; first < count;) {
        const int width = segment_width(slots, count, first);
        if (keep[first] < 1 || keep[first] > width) {
            err = "verify: accepted prefix outside segment"; return false;
        }
        for (int i = first + 1; i < first + width; ++i) if (keep[i] != 0) {
            err = "verify: commit count belongs at the segment start"; return false;
        }
        first += width;
    }
    return true;
}

} // namespace strata::core
