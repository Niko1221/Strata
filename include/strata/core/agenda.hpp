// include/strata/core/agenda.hpp - Delta 4, node D4-7 ("agenda continua"), phase 1.
//
// The pure decisions behind three host-visible pieces of the engine's request agenda.  Nothing here includes
// CUDA and nothing here allocates: the engine calls these functions and tests/core/agenda_test.cpp exercises
// the SAME code host-only, so the rule the engine applies is the rule the test asserts, and a moved threshold
// shows up as a failing test instead of a silent behaviour change.
//
//   1. ADMISSION.  A request's prompt read used to be one blocking `sp.run` when there was no free slot to
//      interleave with, and the loop took the `BGEN` whole before the next window.  The state machine below is
//      the `WAIT_SLOT -> READING -> READY -> ACTIVE` progression: `READING` advances exactly ONE fine chunk
//      (`C_adm`) per turn, so the read is a sequence of turns and the decode window of the slots already
//      active runs FIRST.  The slot just admitted enters the NEXT window - `batch_step` rebuilds its row set
//      from the active slots on every call, so no slot is disturbed and no mid-window insert is needed.
//   2. THE STEP BUDGET IS BY TOKENS, NOT BY TIME.  `STRATA_BATCH_DECODE_SHARE` is a fraction of the chunk's
//      WALL TIME (`generate.cpp` reads it once and spins decode until `share x ms` elapse); it is not a token
//      budget, it never applies inside a chunk, and it does not cover "no active slot".  `agenda_step` below
//      reserves the decode rows first (<= kVerifyMaxT) and hands the prefill chunk the remainder - decode
//      first, in tokens.  The share stays as the compatibility path: when it is set explicitly the engine
//      keeps the base tag's time-share behaviour.
//   3. A FINER `STOP` AND AGING.  `should_stop` is checked by `Prefill::run` before every chunk
//      (`src/prefill/prefill.cpp`), so with a full 8192-token chunk a client that goes away waits up to
//      8192 tokens for the read to notice.  Reading in `C_adm` chunks makes the check land every `C_adm`
//      tokens with no change to the chunk arithmetic (the chunks still partition the same prompt).  AGING is
//      the floor of progress: if a token budget ever leaves the read nothing (`decode_rows >= step_tokens`),
//      the turn still advances (one token at least), and after `age_threshold` such turns in a row the read is
//      granted its full `C_adm` chunk and SKIPS that turn's window - a busy slot cannot starve the read.
//
// OPT-IN.  The engine reads `STRATA_BATCH_AGENDA` (unset/0 = off) and `STRATA_BATCH_DECODE_SHARE` stays the
// switch for the legacy time share.  With the agenda off none of this runs and the engine is byte-for-byte the
// base tag's; with it on, the read is chunked at `C_adm` and the budget is by tokens.  See docs/FLAGS.md
// ("The continuous agenda (STRATA_BATCH_AGENDA)").
//
// NO-CLAIMS (docs/FLAGS.md keeps the same list):
//   * It is NOT a mid-window row insert: the window is rebuilt from the active slots every step, exactly as
//     today, so the verifier's "a window keeps every row" invariant (include/strata/core/verify.hpp) is
//     untouched.  That was the reason phase 2 of the plan marked a mid-window insert optional for N <= 4.
//   * It does NOT change the prompt arithmetic and does NOT touch `batch_step`: the chunks partition the same
//     prompt, so the positions, the residual rows and the per-slot MTP draft are the same; only WHEN the read
//     advances and WHEN a window runs change.
//   * It does NOT raise the slot count: the decode rows are still capped at kVerifyMaxT (8), i.e. 4 slots
//     under `--batch-mtp`.  `--batch` / `parallel` are out of its reach.

#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

// ── 1. The admission chunk and the fineness of the STOP check ────────────────────────────────────────────────

/// The chunk the admission read advances per turn: the full prefill chunk capped at `c_adm` (the fine chunk).
/// `c_adm <= 0` keeps the full chunk (the agenda's own negative control).
inline int64_t agenda_adm_chunk(int64_t chunk, int64_t c_adm) {
    if (c_adm <= 0) return std::max<int64_t>(chunk, 1);
    return std::max<int64_t>(1, std::min(chunk, c_adm));
}

/// How many `should_stop` checks a full prefill chunk of `chunk` tokens now gets: one per admission chunk.
/// 1 is the base tag's granularity (the whole chunk).
inline int64_t agenda_stop_checks(int64_t chunk, int64_t c_adm) {
    if (chunk <= 0) return 0;
    const int64_t c = agenda_adm_chunk(chunk, c_adm);
    return (chunk + c - 1) / c;
}

// ── 2. The decode-first step budget (tokens) ─────────────────────────────────────────────────────────────────

/// One agenda step: the rows the decode windows serve and the prefill chunk that takes the remainder.
struct AgendaStep {
    int decode_rows = 0;        ///< rows the windows of this step serve for the active slots (<= max_rows)
    int decode_windows = 0;     ///< windows that serve them (one window holds up to max_rows rows)
    int64_t prefill_chunk = 0;  ///< tokens the read may advance this step (the budget after the decode rows)
    bool interleave = false;    ///< a decode window runs before the prefill chunk (decode first)
};

/// The step for a step budget of `step_tokens` tokens: the decode rows are reserved FIRST and the prefill
/// chunk takes what is left, capped at `chunk`.  With no active slot (`!any_active`) there is nothing to
/// decode-first: the read gets the whole chunk and no window runs (the base tag's "no slot to interleave").
/// `max_rows` is the verifier's row cap (kVerifyMaxT = 8).  `batch_mtp` makes each active slot cost two rows.
inline AgendaStep agenda_step(bool any_active, int active_slots, bool batch_mtp, int64_t chunk,
                              int64_t step_tokens, int max_rows) {
    AgendaStep s;
    if (!any_active || active_slots <= 0 || max_rows <= 0) {
        s.prefill_chunk = std::max<int64_t>(chunk, 1);
        return s;
    }
    const int per = batch_mtp ? 2 : 1;
    s.decode_rows = std::min<int>(max_rows, active_slots * per);
    s.decode_windows = (s.decode_rows + max_rows - 1) / max_rows;   // one window holds max_rows rows
    s.interleave = s.decode_rows > 0;
    const int64_t remainder = step_tokens - (int64_t) s.decode_rows;   // decode first: reserve, then prefill
    s.prefill_chunk = std::max<int64_t>(0, std::min<int64_t>(std::max<int64_t>(chunk, 1), remainder));
    return s;
}

// ── 3. Aging: the floor of prefill progress ──────────────────────────────────────────────────────────────────

/// True once the read has gone `age_threshold` consecutive turns with no advance.  `age_threshold <= 0` means
/// the aging grant is never forced (a bounded 1-token floor still keeps the loop live - see `agenda_turn`).
inline bool agenda_aging_forces_prefill(int64_t starved_turns, int64_t age_threshold) {
    return age_threshold > 0 && starved_turns >= age_threshold;
}

/// One agenda turn: how many tokens the read advances and whether the decode window runs first.  `starved` is
/// the counter carried in (turns so far with no advance).  With a positive `prefill_chunk` the read advances by
/// it and the counter resets.  With zero, the turn still advances by at least ONE token (the loop never spins),
/// and when aging fires it grants the full admission chunk `c_adm`, resets the counter and DOES NOT run this
/// turn's window (the read gets its turn).
struct AgendaTurn {
    int64_t chunk = 0;
    bool decode_first = false;
    bool aging = false;
    int64_t starved_after = 0;
};

inline AgendaTurn agenda_turn(const AgendaStep& s, int64_t c_adm, int64_t starved, int64_t age_threshold) {
    AgendaTurn t;
    if (s.prefill_chunk > 0) {
        t.chunk = s.prefill_chunk;
        t.decode_first = s.interleave;
        t.starved_after = 0;
        return t;
    }
    const bool age = agenda_aging_forces_prefill(starved, age_threshold);
    t.aging = age;
    t.chunk = age ? std::max<int64_t>(c_adm, 1) : 1;   // aging: the full admission chunk; else a 1-token floor
    t.decode_first = s.interleave && !age;             // aging skips the window: the read goes this turn
    t.starved_after = age ? 0 : starved + 1;
    return t;
}

// ── 4. The admission state machine ───────────────────────────────────────────────────────────────────────────

enum class AdmState { WaitSlot, Reading, Ready, Active };

inline const char* adm_state_name(AdmState s) {
    switch (s) {
        case AdmState::WaitSlot: return "wait_slot";
        case AdmState::Reading:  return "reading";
        case AdmState::Ready:    return "ready";
        case AdmState::Active:   return "active";
    }
    return "?";
}

/// One admission: the slot it will continue in, the prompt it must read, the cursor, and its state.
/// `WAIT_SLOT` is before the engine holds the slot; `begin()` takes it to `READING`; every `advance_chunk`
/// moves the cursor by exactly one admission chunk; reaching `total` takes it to `READY` (the first token is
/// produced by the caller); `activate()` takes it to `ACTIVE` once the slot owns the session.
struct Admission {
    int slot = -1;
    int64_t max_new = 0;
    int64_t total = 0;      ///< prompt tokens to read
    int64_t read_to = 0;    ///< the cursor
    AdmState state = AdmState::WaitSlot;

    /// WAIT_SLOT -> READING.  Idempotent: from any other state it does nothing.
    void begin() { if (state == AdmState::WaitSlot) state = AdmState::Reading; }

    /// READING -> one more admission chunk: returns the tokens to read this turn (0 when not READING).
    /// The last step clamps to the remaining tokens and takes the state to READY.
    int64_t advance_chunk(int64_t c_adm) {
        if (state != AdmState::Reading) return 0;
        const int64_t remaining = std::max<int64_t>(total - read_to, 0);
        const int64_t step = std::min<int64_t>(std::max<int64_t>(c_adm, 1), remaining);
        read_to += step;
        if (read_to >= total) state = AdmState::Ready;
        return step;
    }

    /// READY -> ACTIVE (the slot's session is set); from any other state it does nothing.
    void activate() { if (state == AdmState::Ready) state = AdmState::Active; }

    bool reading() const { return state == AdmState::Reading; }
    bool done() const { return state == AdmState::Ready || state == AdmState::Active; }
};

}  // namespace strata::core
