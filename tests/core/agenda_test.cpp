// tests/core/agenda_test.cpp - Delta 4 / D4-7 ("agenda continua"), phase 1: the four pure decisions the agenda
// adds, exercised host-only (no CUDA, no GPU, no model):
//
//   1. The admission state machine advances exactly ONE fine chunk per turn (WAIT_SLOT -> READING -> READY ->
//      ACTIVE) instead of running the whole prompt at once, so the read is not a blocking run.
//   2. The step budget is decode-first BY TOKENS: the decode rows are reserved first and the window never runs
//      out of rows; the prefill takes the remainder.
//   3. The STOP granularity drops from the full prefill chunk to the admission chunk C_adm.
//   4. Aging bounds a starvation the token budget could cause, and with the switch off every decision is the
//      base tag's (the negative control the delta asks for: `c_adm <= 0` == the full chunk, `!any_active` == no
//      interleave).
//
// The test links nothing but the header: it is built by STRATA_BUILD_CONVERSATION_TESTS alongside the other
// host-only tests (see _host_build.bat).  It mirrors include/strata/core/agenda.hpp the way
// stage_plan_test.cpp mirrors stage_plan.hpp: the rule the engine applies is the rule the test asserts.
#include "strata/core/agenda.hpp"

#include <cstdio>
#include <string>

using strata::core::Admission;
using strata::core::AdmState;
using strata::core::adm_state_name;
using strata::core::agenda_adm_chunk;
using strata::core::agenda_aging_forces_prefill;
using strata::core::agenda_step;
using strata::core::agenda_stop_checks;
using strata::core::agenda_turn;

namespace {

int g_fail = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::string i2s(long long v) { return std::to_string(v); }

}  // namespace

int main() {
    // ── 1. The admission state machine: one fine chunk per turn, not the whole prompt ───────────────────────
    std::printf("agenda: the admission state machine (one fine chunk per turn)\n");
    {
        Admission a;
        a.slot = 2;
        a.total = 1300;   // a prompt longer than the admission chunk
        check(a.state == AdmState::WaitSlot && std::string(adm_state_name(a.state)) == "wait_slot",
              "a fresh admission waits for its slot");
        check(a.advance_chunk(512) == 0, "WAIT_SLOT reads nothing before begin() (no blocking)");
        a.begin();
        check(a.reading() && std::string(adm_state_name(a.state)) == "reading", "begin(): WAIT_SLOT -> READING");
        a.begin();
        check(a.reading(), "begin() is idempotent (a second call keeps READING)");
        // ONE turn advances exactly one admission chunk - not the whole prompt
        const int64_t t1 = a.advance_chunk(512);
        check(t1 == 512 && a.read_to == 512 && !a.done(),
              "turn 1 advances exactly one chunk (512 of 1300), the admission is still reading");
        check(a.advance_chunk(512) == 512 && a.read_to == 1024, "turn 2 advances one more chunk (1024)");
        const int64_t t3 = a.advance_chunk(512);
        check(t3 == 276 && a.read_to == 1300, "turn 3 clamps to what is left (276) and reaches the prompt end");
        check(a.state == AdmState::Ready && a.done() && !a.reading(),
              "READING -> READY once the cursor reaches the prompt end (the first token is the caller's)");
        check(a.advance_chunk(512) == 0, "READY reads nothing more");
        a.activate();
        check(a.state == AdmState::Active && std::string(adm_state_name(a.state)) == "active",
              "activate(): READY -> ACTIVE (the slot owns the session)");
        check(a.advance_chunk(512) == 0, "ACTIVE is terminal: no further reads");
    }
    {
        // a prompt shorter than one chunk ends in a single turn (never a spin), and a zero chunk means the
        // whole chunk (the caller resolved C_adm already; 0 is the base tag's granularity)
        Admission b;
        b.total = 100; b.begin();
        check(b.advance_chunk(512) == 100 && b.state == AdmState::Ready,
              "a prompt shorter than C_adm ends in one clamped turn");
        Admission c;
        c.total = 8; c.begin();
        check(c.advance_chunk(0) == 1, "a zero C_adm never reads 0 tokens (a 1-token floor keeps the loop live)");
    }

    // ── 2. The decode-first token budget: the window never runs out of rows ──────────────────────────────────
    std::printf("agenda: the decode-first token budget (the window keeps its rows)\n");
    {
        const int MAX = 8;
        // 4 slots under --batch-mtp cost 8 rows: one window, and the prefill takes the remainder of the budget
        const auto s4 = agenda_step(true, 4, true, 512, 520, MAX);
        check(s4.decode_rows == 8, "4 slots x 2 rows (--batch-mtp): the window reserves 8 rows (kVerifyMaxT)");
        check(s4.decode_windows == 1, "8 rows fit one window (kVerifyMaxT): one window per step");
        check(s4.interleave, "an active slot means the window runs first");
        check(s4.prefill_chunk == 512, "with a 520-token budget the prefill still gets its full 512-token chunk");
        // two plain slots
        const auto s2 = agenda_step(true, 2, false, 512, 520, MAX);
        check(s2.decode_rows == 2 && s2.decode_windows == 1 && s2.prefill_chunk == 512,
              "2 plain slots: 2 rows, one window, the prefill keeps the chunk");
        // the decode rows come FIRST: a budget of exactly the decode rows leaves the prefill nothing
        const auto s0 = agenda_step(true, 4, true, 512, 8, MAX);
        check(s0.decode_rows == 8 && s0.prefill_chunk == 0,
              "a budget of exactly the decode rows: the prefill gets 0 (decode first)");
        // more rows than one window holds: the step schedules the extra windows, it does not drop rows
        const auto smany = agenda_step(true, 5, true, 512, 520, MAX);
        check(smany.decode_rows == 8 && smany.decode_windows == 1,
              "5 slots x 2 = 10 requested rows cap at kVerifyMaxT (8) in one window (the --batch 4 ceiling)");
        const auto s3 = agenda_step(true, 3, true, 512, 520, MAX);
        check(s3.decode_rows == 6 && s3.decode_windows == 1, "3 slots x 2 rows: 6 rows, one window");
    }
    {
        // NEGATIVE CONTROL: no active slot == the base tag's read (whole chunk, no window)
        const auto s = agenda_step(false, 0, true, 512, 520, 8);
        check(!s.interleave && s.decode_rows == 0 && s.prefill_chunk == 512,
              "no active slot: no interleave, the read gets the whole chunk (base tag)");
        const auto sm = agenda_step(true, 4, true, 512, 520, 0);
        check(!sm.interleave && sm.prefill_chunk == 512,
              "a zero row cap: no window to run, the read gets the whole chunk");
    }

    // ── 3. The STOP granularity: from the full chunk down to C_adm ───────────────────────────────────────────
    std::printf("agenda: the STOP granularity (the full chunk vs C_adm)\n");
    check(agenda_stop_checks(8192, 512) == 16, "an 8192-token chunk at C_adm 512: 16 STOP checks (was 1)");
    check(agenda_stop_checks(8192, 1024) == 8, "at C_adm 1024: 8 checks");
    check(agenda_stop_checks(8192, 8192) == 1, "C_adm == the chunk: 1 check (the base tag's granularity)");
    check(agenda_stop_checks(8192, 0) == 1, "the switch off (C_adm 0): the full chunk, 1 check (unchanged)");
    check(agenda_stop_checks(8192, 20000) == 1, "C_adm above the chunk is clamped to the chunk: 1 check");
    check(agenda_adm_chunk(8192, 512) == 512, "the admission chunk is 512");
    check(agenda_adm_chunk(300, 512) == 300, "a chunk below C_adm is not raised above it");
    check(agenda_adm_chunk(8192, 0) == 8192, "NEGATIVE CONTROL: C_adm 0 keeps the full chunk (base tag)");
    check(agenda_adm_chunk(0, 512) == 1, "a zero chunk still yields a positive admission chunk (never 0)");

    // ── 4. Aging: the floor of prefill progress ──────────────────────────────────────────────────────────────
    std::printf("agenda: aging (the floor of prefill progress)\n");
    {
        // a budget that leaves the prefill nothing: 8 decode rows against an 8-token budget
        const auto s = agenda_step(true, 4, true, 512, 8, 8);
        check(s.prefill_chunk == 0, "the starved step: the budget leaves the prefill 0 tokens");
        // turns 0..3 (threshold 4): the 1-token floor keeps the read live, and the window still runs first
        int64_t starved = 0;
        for (int64_t i = 0; i < 4; ++i) {
            const auto t = agenda_turn(s, 512, starved, 4);
            check(t.chunk == 1 && !t.aging && t.decode_first,
                  "starved turn " + i2s(i) + ": a 1-token floor, the window still runs first");
            starved = t.starved_after;
        }
        check(starved == 4, "four starved turns were counted");
        // the fifth turn: aging grants the full admission chunk and SKIPS the window
        const auto t = agenda_turn(s, 512, starved, 4);
        check(t.aging && t.chunk == 512, "aging fires: the read is granted its full 512-token chunk");
        check(!t.decode_first, "the aging turn runs NO window: the read gets its turn");
        check(t.starved_after == 0, "the starvation counter resets after the grant");
        // a non-starved turn resets the counter and takes the budget's chunk
        const auto good = agenda_step(true, 2, false, 512, 520, 8);
        const auto g = agenda_turn(good, 512, /*starved=*/99, 4);
        check(g.chunk == 512 && !g.aging && g.decode_first && g.starved_after == 0,
              "a budget with room resets the counter and never needs aging");
        // age 0 = aging off: the grant is never forced, but the 1-token floor still bounds the turn
        const auto off = agenda_turn(s, 512, /*starved=*/1000, /*age_threshold=*/0);
        check(!off.aging && off.chunk == 1, "age 0: aging never fires, the 1-token floor still keeps the read live");
        check(!agenda_aging_forces_prefill(999, 0), "age 0: agenda_aging_forces_prefill is always false");
        check(agenda_aging_forces_prefill(4, 4) && !agenda_aging_forces_prefill(3, 4),
              "the threshold is exact: it fires at age_threshold, not before");
    }

    // ── The switch off: the base tag's decisions, byte for byte ──────────────────────────────────────────────
    std::printf("agenda: with the switch off the decisions are the base tag's\n");
    check(agenda_adm_chunk(8192, /*c_adm=*/0) == 8192 && agenda_stop_checks(8192, 0) == 1,
          "C_adm 0: the whole chunk and one STOP check == today");
    {
        const auto nb = agenda_step(false, 0, false, 8192, 0, 8);
        check(!nb.interleave && nb.prefill_chunk == 8192 && nb.decode_rows == 0,
              "no active slot, no budget: the read is one whole chunk with no window == today");
    }

    std::printf("agenda: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
