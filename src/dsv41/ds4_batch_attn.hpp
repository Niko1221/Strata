// Strata. One step's attention tables for several sequences sharing the window buffer and the compressed rows
// (docs/DSV41_BATCHING.md, M2 - "M2, re-scoped").
//
// The attention kernel already runs T queries in one launch, each with its own window offset and its own compressed rows
// (ds4_kernels.cuh's AttnRows: win0[t], nwin[t], nsel[t], and the selection table with a per-query stride).  What M2 has
// to build is that table when the T queries are T different sequences, each living in its own region of the one window
// buffer and its own rows of the one compressed-KV table: the sequence's window ring at win_base for "win" rows, its
// compressed rows from kv_base on.
//
// This header is the bookkeeping only - no GPU, no model - so the arithmetic that decides which rows a step reads can be
// checked without one (a row counted wrong does not crash, it answers subtly wrong).
#pragma once

#include <cstdint>
#include <vector>

namespace strata {
namespace ds4 {
namespace batchattn {

/// Where one sequence's state lives in the shared buffers.
struct Layout {
    int64_t pos = 0;    // the tokens it holds: min(pos, win) window rows are valid
    int win_base = 0;   // its first row in the window buffer (its own ring of "win" rows)
    int kv_base = 0;    // its first row in the compressed table (a source's)
    int kv_rows = 0;    // how many compressed rows its region holds (the cap)
};

/// A step: n sequences decoding one token each, the ring size, the compression ratio and the row cap a position selects.
struct Step {
    int n = 0;
    Layout seq[8];
    int win = 0;
    int ratio = 1;
    int kcap = 0;
};

/// The window rows before a sequence's position: the ring is full at "win", and a sequence never reaches into another's
/// region because its base is its own.
inline int window_rows(const Step& s, int t) {
    const int64_t n = s.seq[t].pos < (int64_t) s.win ? s.seq[t].pos : (int64_t) s.win;
    return (int) n;
}

/// The compressed rows a sequence has *written* at a position: pos / ratio (slot_save's count - a compressor's group
/// completes every ratio tokens).  It is one less than selectable() at a group's boundary, which is the off-by-one that
/// makes a batched answer subtly wrong instead of crashing.
inline int written_rows(int64_t pos, int ratio) { return ratio > 0 ? (int) (pos / ratio) : 0; }

/// The row a position's group lands in when the compressor writes it: pos / ratio (the same row it is written into).
inline int row_of(int64_t pos, int ratio) { return written_rows(pos, ratio); }

/// The row a sequence's compressor writes a position's group into: its region's start (kv_base) plus row_of().  One
/// sequence writes at row_of(pos, ratio) - kv_base 0 - so this is the address the model's finish step gets either way.
inline int write_row(const Layout& l, int64_t pos, int ratio) { return l.kv_base + row_of(pos, ratio); }

/// The compressed rows its position can select: (pos + 1) / ratio, capped by the region and by kcap.
inline int selectable(const Step& s, int t) {
    int64_t n = s.ratio > 0 ? (s.seq[t].pos + 1) / s.ratio : 0;
    if (s.kcap > 0 && n > s.kcap) n = s.kcap;
    if (n > s.seq[t].kv_rows) n = s.seq[t].kv_rows;
    return (int) n;
}

/// Fill the step's AttnRows fields: win0[t] is the sequence's first window row, nwin[t] its valid rows, nsel[t] how many
/// compressed rows it selects.  False when n is out of the table or a sequence has no window row to read.
inline bool rows(const Step& s, int* win0, int* nwin, int* nsel, int max_t) {
    if (s.n <= 0 || s.n > max_t || s.n > 8) return false;
    for (int t = 0; t < s.n; ++t) {
        const int w = window_rows(s, t);
        if (w <= 0) return false;
        win0[t] = s.seq[t].win_base;
        nwin[t] = w;
        nsel[t] = selectable(s, t);
    }
    return true;
}

/// The indexer's picks for sequence t (row ids relative to its own region) into the shared table: its kv_base added, and
/// refused when a pick is outside the sequence's region (the answer would read another conversation's rows).
inline bool place(const Step& s, int t, const int* picks, int n, std::vector<int>& out) {
    if (t < 0 || t >= s.n || n < 0) return false;
    if (picks != nullptr && n > selectable(s, t)) return false;
    out.clear();
    out.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        const int r = picks[i];
        if (r < 0 || r >= s.seq[t].kv_rows) return false;
        out.push_back(s.seq[t].kv_base + r);
    }
    return true;
}

/// Whether the sequences' regions are disjoint - what makes one buffer serve them (false: two would read each other).
inline bool disjoint(const Step& s) {
    for (int i = 0; i < s.n; ++i) {
        for (int j = i + 1; j < s.n; ++j) {
            if (s.seq[i].win_base < s.seq[j].win_base + s.win && s.seq[j].win_base < s.seq[i].win_base + s.win) return false;
            const int64_t i0 = s.seq[i].kv_base, i1 = i0 + s.seq[i].kv_rows;
            const int64_t j0 = s.seq[j].kv_base, j1 = j0 + s.seq[j].kv_rows;
            if (i0 < j1 && j0 < i1) return false;
        }
    }
    return true;
}

}  // namespace batchattn
}  // namespace ds4
}  // namespace strata
