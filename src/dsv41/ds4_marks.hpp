// src/dsv41/ds4_marks.hpp - maya-ds4's restore points: which one a new prompt goes back to, which of the ones a request
// asks for its read keeps, and which one gives way to a new one.  A restore point is the state at a position (the
// window rings and the compressors' partial groups; the rest of the state is indexed by position).  Besides the three a
// conversation always has (the end of its last prompt, the position before it, the anchor), the server may ask for more
// on a GEN line (marks=<pos>,... and shared=<pos>): the start of the last user message, a point near the prompt's tail,
// the end of a start that other conversations share (a system prompt and its tools).  The read stops at each to keep
// it, so a prompt that differs from the conversation after one of them - its last message edited, a new question
// after the same document, a history rendered again - goes back there instead of further.  Pure logic, no GPU
// (tests/ds4_marks_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::ds4 {

// the restore points a conversation keeps besides its three, at most (MAYA_DS4_RESTORE_POINTS is clamped to it)
constexpr int kMarksMax = 16;

// the latest of the points `at` (positions; -1: none) at or before `limit` - where the state's tokens still agree with
// the new prompt - and at or before the state's own position `pos`; the first of equal ones; -1: none
inline int latest_point(const std::vector<int64_t>& at, int64_t limit, int64_t pos) {
    int b = -1;
    for (size_t i = 0; i < at.size(); ++i)
        if (at[i] >= 0 && at[i] <= limit && at[i] <= pos && (b < 0 || at[i] > at[(size_t) b])) b = (int) i;
    return b;
}

// a restore point a prompt's read keeps: where, and whether it is the shared start (kept before the others)
struct MarkAt {
    int64_t pos = 0;
    bool shared = false;
};

// The restore points the read of a prompt's tokens [from, total) keeps, in order, from the positions a request asks
// for (`asked`, and `shared`: -1 none): those the read starts at or goes past - from `from` (the state there, kept
// before the read) to short of the prompt's last token (checkpoint 1 keeps the state before it).  One inside a
// picture's span (`spans`: start, length) is moved to the span's start (a span is read whole); one fewer than `gap`
// tokens past `from` is kept at `from` (no read of its own for a few tokens), and of two fewer than `gap` apart the
// first is kept, with the second's role too.  At most `cap`: the shared start first, then the latest.
inline std::vector<MarkAt> mark_plan(const std::vector<int64_t>& asked, int64_t shared, int64_t from, int64_t total,
                                     int cap, int64_t gap,
                                     const std::vector<std::pair<int64_t, int64_t>>& spans = {}) {
    std::vector<MarkAt> v;
    if (cap <= 0 || from < 0 || total - from < 2) return v;
    const auto add = [&](int64_t p, bool sh) {
        for (const auto& s : spans)
            if (p > s.first && p < s.first + s.second) p = s.first;
        if (p < from || p >= total - 1) return;
        if (p - from < gap) p = from;
        v.push_back({p, sh});
    };
    for (int64_t p : asked) add(p, false);
    if (shared >= 0) add(shared, true);
    std::stable_sort(v.begin(), v.end(), [](const MarkAt& a, const MarkAt& b) { return a.pos < b.pos; });
    std::vector<MarkAt> out;
    for (const MarkAt& m : v) {
        if (!out.empty() && m.pos - out.back().pos < std::max<int64_t>(gap, 1)) {
            out.back().shared = out.back().shared || m.shared;
            continue;
        }
        out.push_back(m);
    }
    while ((int) out.size() > cap) {   // (the earliest that is not the shared start goes first)
        auto it = std::find_if(out.begin(), out.end(), [](const MarkAt& m) { return !m.shared; });
        out.erase(it == out.end() ? out.begin() : it);
    }
    return out;
}

// A learned start (STRATA_DS4_LEARN_STARTS).  The shared start a request names ends with the system prompt and its
// tools, but a new conversation often shares more with an earlier one: an agent that opens every session with the same
// instructions in its first user message parts from the last session only where its own first words begin.  `parts` is
// the first token at which the prompt differs from the furthest of what the engine knows (the conversation it sets
// aside, the kept conversations, the kept starts).  When the read of tokens [from, total) starts at least `least`
// tokens before the point `back` tokens short of that place (the tokenizer may merge across it), the read keeps one
// more restore point there, and the serve loop keeps the state at it apart like a shared start, so the next prompt
// that begins the same way reads only what follows.  `cuts` are the points the read keeps already (mark_plan's: in
// order, a gap apart):
//  - one within the gap at or before the learned point stands in for it (nothing is added);
//  - one within the gap after it does not - it may lie past the place the prompt parts, and a start that holds this
//    prompt's own words is no start for the next one: the learned point moves back to stay a gap short of it;
//  - none is added ahead of a shared start (the serve loop keeps that one apart only as the read's first point), past
//    the cap, fewer than a gap past `from`, at the prompt's last token, or where the start would be shorter than
//    `floor` (the shortest start kept).
// Returns the position (one of `cuts` then, added in order if it was not there), or -1: none.
inline int64_t learn_point(std::vector<MarkAt>& cuts, int64_t parts, int64_t from, int64_t total, int64_t back,
                           int64_t least, int64_t floor, int cap, int64_t gap = 16) {
    if (cap <= 0 || parts - back - from < least || parts - back < floor) return -1;
    int64_t at = parts - back;
    const auto stand_in = [&]() {
        for (const MarkAt& c : cuts)
            if (c.pos <= at && at - c.pos < gap) {
                at = c.pos;
                return true;
            }
        return false;
    };
    bool have = stand_in();
    if (!have) {
        for (const MarkAt& c : cuts)   // (in order: the first one after `at` is the nearest)
            if (c.pos > at && c.pos - at < gap) {
                at = c.pos - gap;
                break;
            }
        have = stand_in();
    }
    if (at - from < gap || at >= total - 1) return -1;
    if (!have) {
        for (const MarkAt& c : cuts)
            if (c.shared && c.pos > at) return -1;
        if ((int) cuts.size() >= cap) return -1;
        cuts.insert(std::upper_bound(cuts.begin(), cuts.end(), at, [](int64_t p, const MarkAt& c) { return p < c.pos; }),
                    MarkAt{at, false});
    }
    return at;
}

// The tail point's cost on this PC (the server's tail=<pos>, MAYA_DS4_RESTORE_TAIL; #668).  The read stops there, so
// the tokens after it are a run of their own: where most experts are not in VRAM, that short run costs nearly a whole
// pass over them (each expert's weights read for a few rows - on the CPU lane, or lent to the GPU's prompt path one
// run at a time; on a layer split whose experts stream, one more pass of them through every stage, one stage after
// another), seconds on a long prompt that a single run did not spend.  Measured on the reads that kept the point, in
// wall time a token: the run after it against the runs before it.  Read as one run, the tail's tokens would have cost
// about the long read's time a token (the long-prompt skip's cut moves with the run's end, and a split's pipeline fills
// and drains once either way), so the point costs what its run spends beyond that.  It is kept while that is no more
// than reading its tokens again: a ratio of 2 or less.  (v1.1.39 compared the time a routed expert pick took, over the
// split's stages: a run's fixed cost - the experts streamed again, the stages one after another - hid behind both, and
// a 4-GPU split kept a point that cost 4-9x, #668.)  Until two reads have measured it the point is kept, and one tail
// point in `probe` is kept while it is left out, to measure again (the cost moves with the experts the PC holds in
// VRAM).  Pure logic (tests/ds4_marks_test.cpp).
struct TailCost {
    int n = 0;              // the reads measured
    double ratio = 0.0;     // their tail run's time a token over the rest of the read's, a running mean (the last two
                            // halves: the first read after a start is slow everywhere)
    int left_out = 0;       // tail points left out since the last one kept
    double last_tail_ms = 0.0, last_main_ms_token = 0.0;   // the last read measured (the log's line)
    size_t last_tail_tokens = 0;
    static constexpr double kKeepAt = 2.0;
    // a read that kept the point: its runs before it (ms, tokens) and its run after it (ms, tokens).  A read too short
    // before the point to be a long run's rate (`tail_tokens` x 4 tokens or fewer) is not counted: false
    bool add(double ms_main, size_t main_tokens, double ms_tail, size_t tail_tokens) {
        if (!(ms_main > 0.0) || !(ms_tail > 0.0) || tail_tokens == 0 || main_tokens <= 4 * tail_tokens) return false;
        const double r = (ms_tail / (double) tail_tokens) / (ms_main / (double) main_tokens);
        last_tail_ms = ms_tail;
        last_tail_tokens = tail_tokens;
        last_main_ms_token = ms_main / (double) main_tokens;
        ratio = n == 0 ? r : 0.5 * (ratio + r);
        ++n;
        left_out = 0;
        return true;
    }
    bool measured() const { return n >= 2; }
    bool costly() const { return measured() && ratio > kKeepAt; }
    // keep the next tail point?  (a point left out counts towards the next probe)
    bool keep(int probe = 32) {
        if (!costly()) return true;
        if (++left_out >= probe) {
            left_out = 0;
            return true;
        }
        return false;
    }
};

// a kept restore point as the choice of the one to give up sees it
struct MarkUse {
    int64_t pos = 0;
    bool shared = false;
    uint64_t used = 0;   // (its last use: kept or gone back to - a clock that only grows)
};

// the restore point that gives way to a new one when a conversation holds its cap: the least recently used of those
// that are not a shared start, else the least recently used; -1: none held
inline int mark_victim(const std::vector<MarkUse>& v) {
    for (int pass = 0; pass < 2; ++pass) {
        int b = -1;
        for (size_t i = 0; i < v.size(); ++i)
            if ((pass == 1 || !v[i].shared) && (b < 0 || v[i].used < v[(size_t) b].used)) b = (int) i;
        if (b >= 0) return b;
    }
    return -1;
}

}  // namespace strata::ds4
