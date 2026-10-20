// The pipelined decode loop's scheduler as a host state machine (--pipeline-windows 2 and 3 on a layer split).
//
// --pipeline-windows on a layer split: the oldest window not verified (win[0]) runs the last stage while the windows
// behind it (guesses of how it ends) run the front stages, each one stage behind the window before it.  This file
// decides WHAT happens and in which order; it calls no CUDA.  The engine (generate.cpp) implements `PlIo`: the
// polls read its verifiers and drafter, and `exec` turns each abstract operation into the verifier / drafter /
// snapshot call it names.  pl_sched_test drives the same state machine against a symbolic GPU.
//
// Three modes:
//   N2  pw 2, global undo, 2 slots (slot = seq & 1): today's loop (generate.cpp, the pw-2 loop) operation for
//       operation, so the engine's translation can be checked against the old loop event by event.
//   B0  pw 2, per-stage undo, 3 slots: a wrong guess is undone on each front stage as soon as no doomed window runs
//       there, and the fresh window goes on without waiting for the others.
//   B   pw 3 (a third window in flight), per-stage undo, 4 slots, the chain's outputs up to 11, the chain stopped at
//       the point that decides its last new window (chain_stop).
//
// Slots: a window takes one slot - a verifier on every stage, a hand-off per boundary, a plan sink, a
// GDN snapshot per front stage - from a free list the first time it needs one (prestage or launch), and gives it back
// once nothing may still read it: a doomed window when it has stopped; a verified window when every commit of its
// (the last stage's, the front stages', an undo's replays) is queued and the chain read from its last-stage rows is
// launched.  Every reply token stays the serial loop's: only the oldest window is emitted, a wrong guess
// is undone per stage by restoring the verified window's snapshot and replaying its commit with the real count.
#pragma once

#include <cstdint>
#include <string>

namespace strata::pl {

inline constexpr int kMaxSlots = 5;      // STRATA_PIPELINE_SLOTS at most
inline constexpr int kMaxFront = 8;      // front stages (every stage but the last)
inline constexpr int kChainMax = 11;     // the drafter's chain outputs at most (MtpDrafter::kChainMax)
inline constexpr int kMaxRows = 8;       // a window's rows at most (kVerifyMaxT)
inline constexpr int kMaxWin = 3;        // windows not verified at most (depth 3)

enum class Mode : uint8_t { N2 = 0, B0 = 1, B = 2 };
const char* mode_name(Mode m);

struct Config {
    Mode mode = Mode::N2;
    int n_front = 3;               // pl_nf: front stages (n_stages - 1)
    int slots = 0;                 // NS; 0: by mode (N2 2, B0 3, B 4); N2 always 2
    int S = 6;                     // the verifier's rows (a lookup window up to S)
    int S_mtp = 4;                 // the drafter's windows
    double spec_min_p = 0.0;       // req_spec_min_p (<= 0: windows of S_mtp rows)
    float theta = 0.2f;            // the gate of a window behind one guess (STRATA_PIPELINE_THETA)
    float theta3 = -1.0f;          // ... behind two guesses (STRATA_PIPELINE_THETA3; < 0: theta)
    int64_t max_context = 0;
    int chain_out2 = 7;            // the fresh chain's outputs at depth 2 (kVerifyMaxT - 1, today)
    int chain_max = 11;            // depth 3: the chain's outputs at most (STRATA_MTP_CHAIN_MAX)
    bool chain_stop = true;        // depth 3: queue the chain only as far as it is wanted (STRATA_MTP_CHAIN_STOP)
    int force_miss = 0;            // STRATA_PIPELINE_FORCE_MISS: every k-th window behind one guess, row 0 flipped
    int force_miss_c = 0;          // STRATA_PIPELINE_FORCE_MISS_C: the same for windows behind two guesses
    bool prestage = true;          // STRATA_PIPELINE_PRESTAGE
    bool agree = true;             // STRATA_PIPELINE_AGREE
    bool suffix = false;           // o.suffix_draft > 0 (lookup windows)
    float lookup_pon = 0.75f;      // STRATA_PIPELINE_LOOKUP_PON
    int spec_depth = 1 << 20;      // STRATA_PIPELINE_SPEC_DEPTH (debug)

    int depth() const { return mode == Mode::B ? 3 : 2; }
    int ns() const { return mode == Mode::N2 ? 2 : slots > 0 ? slots : mode == Mode::B0 ? 3 : 4; }
    float gate(int k) const { return k >= 2 && theta3 >= 0.0f ? theta3 : theta; }
    bool per_stage() const { return mode != Mode::N2; }
};

// One window (today's PW, generate.cpp, plus the slot pool's and depth 3's fields)
struct PW {
    int T = 0, seq = 0;
    int64_t p = 0;
    int32_t tok[kMaxRows] = {};
    float prob[kMaxRows] = {};     // prob[i]: the drafter's probability of tok[i + 1]
    float p_on = 0.0f;             // the estimate that the window before is accepted whole and tok[0] is its bonus
    // launched: on stage 0; finished: done on every front stage; s1: on the last stage
    bool ready = false, launched = false, finished = false, s1 = false, s1_done = false;
    int fr = 0;                    // front stages done
    bool fl = false;               // in flight on front stage `fr`
    uint32_t com = 0;              // the front stages it was committed on speculatively (bit per stage)
    bool spec = false;             // launched speculatively (behind an unverified window)
    bool made = false;             // made from a chain (or a lookup) as a guess
    bool sfx = false;              // a lookup window (the suffix drafter's drafts)
    int sfx_match = 0;
    // the slot pool and depth 3
    int slot = -1;                 // -1: none yet
    int32_t ple_in[2] = {-1, -1};  // ss.ple_prev when it launches on stage 0 (= its undo value)
    uint8_t lvl = 0;               // 0 fresh, 1 behind one guess, 2 behind two (when made)
    bool prestaged = false, doomed = false;
    bool lk = false;               // made from a lookup (a lookup window's continuation)
    bool gated = false;            // held at stage 0 by the gate or the context (never launched)
    uint64_t uid = 0;              // unique per request (ownership in the symbolic test)
    double t_slot_wait = -1.0;     // when it first found no free slot (statistics)
};

struct UndoJob {
    bool active = false;
    uint64_t uid = 0;
    int slot = -1, seq = 0, keep = 0;
    int64_t p = 0;
    int32_t ple_in[2] = {-1, -1};
    uint32_t pending = 0;          // front stages still to restore and replay
    int n_doomed = 0;
    double t0 = 0.0;
};

enum class Op : uint8_t {
    Gap,          // --adapt-async: a_gap(st == 0 ? 0 : 1) - the stage idles until the launch that follows
    Prestage,     // VF(0, slot).prestage(T, tok, p, ple)
    SnapTake,     // snap_take(st, slot): stage st's GDN state into buf[slot] on its side stream, after the work queued
    SnapWait,     // stage st's stream waits for buf[slot]'s copy (ev[slot])
    Launch,       // VF(st, slot).pl_launch(T, tok, p)  (st == n_front: the last stage, V1(slot))
    SrcWait,      // the last stage's stream waits for slot's source_done (the chain has copied its final_R)
    Commit,       // VF(st, slot).pl_commit_async(keep)  (st == n_front: V1(slot)); spec: keep = T, ahead of a verdict
    Restore,      // stage st's GDN state <- buf[slot], on its stream (an undo)
    PleSet,       // ss.ple_prev = ple (an undo on stage 0, before its replayed commit)
    ChainWaitIdle,// N2 (today's verdict): the host waits until the drafter is idle
    ChainLaunch,  // set_source_R(V1(slot).final_R(0)); chain_launch(T, outp, p, a, force, n_force, n_out, n_early,
                  // n_queue) (B0/B: source_done recorded after its source copy)
    ChainExtend,  // chain_extend(k)
    ChainClose,   // chain_close()
    AdaptVerdict, // pl_adapt() (the blocking tier, without --adapt-async)
    AdaptTick,    // adapt_tick(false) (--adapt-async), once the verdict is done
};
const char* op_name(Op o);

struct PlOp {
    Op op = Op::Gap;
    int st = -1;                   // the stage (n_front: the last)
    int slot = -1;
    int seq = 0;
    uint64_t uid = 0;              // the window it concerns (Commit/Restore: the one whose state it commits/restores)
    int T = 0, keep = 0;
    int64_t p = 0;
    bool spec = false;
    const int32_t* tok = nullptr;  // Prestage/Launch: T tokens
    int32_t ple[2] = {-1, -1};     // Prestage/PleSet
    // ChainLaunch (tokens: the verified window's picks), ChainExtend (k)
    int a = 0, k = 0;
    const int32_t* outp = nullptr;
    int32_t force[16] = {};
    int n_force = 0, n_out = 0, n_early = 0, n_queue = 0, kind = 0;
};

// The suffix drafter's proposal (pick_lookup)
struct Lookup {
    int k = 0;                     // tokens proposed (0: none)
    int match = 0;
    int32_t tok[16] = {};
};

// What the scheduler asks the engine.  Every call is non-blocking except exec (ChainWaitIdle, a Commit's spin on its
// verifier's previous commit) and verdict.
class PlIo {
public:
    virtual ~PlIo() = default;
    // the window in flight on front stage st's verifier `slot`: service it, and once done pl_finish it: 1 done (now
    // finished), 0 running, -1 an error
    virtual int front_poll(int st, int slot, bool doomed) = 0;
    // the same on the last stage; once done, its picks into outp (T of them)
    virtual int last_poll(int slot, int32_t* outp) = 0;
    virtual int chain_ready() = 0;           // chain_outputs_ready (-1: an error)
    virtual int chain_poll() = 0;            // 1 done, 0 running (or open), -1 an error
    virtual bool chain_live() = 0;
    // The chain launched last, as MtpDrafter reports it: chain_cap the outputs it may queue at most (n_out after the
    // cell clamp), chain_n the outputs queued so far (= chain_cap for a chain queued whole; with n_queue the steps
    // queued up to now, and once a closed open chain is done the outputs it really has)
    virtual int chain_cap() = 0;
    virtual int chain_n() = 0;
    virtual const int32_t* chain_tok() = 0;
    virtual const float* chain_prob() = 0;
    virtual void ple_prev(int32_t out[2]) = 0;   // ss.ple_prev now
    virtual int lookup_propose(int max_k, Lookup& out) = 0;   // sfx.propose (k), sfx.last_match
    virtual bool lookup_choose(int T, int k, int match, int& t) = 0;   // policy.choose: true = take the lookup (t rows)
    virtual bool exec(const PlOp& op) = 0;   // false: an error (the engine has said why)
    // emit the window's picks outp[0..a] (up to max_new, EOS); `last` = EOS, max_new or a cancel
    virtual bool verdict(const PW& w, int a, const int32_t* outp, bool& last) = 0;
    virtual void observe(const PW& w, int a, double ms) = 0;   // policy.observe (not for seq 0 or the last)
    virtual double now_ms() = 0;
    virtual void trace(const char* ev, int seq, int x0, int x1) = 0;
};

// The statistics the timing lines print (the first three as today; the fourth is new)
struct Stats {
    int64_t windows = 0, spec = 0, on = 0, undo = 0, gate = 0, fm = 0, fm_c = 0, disagree = 0, late = 0;
    double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
    int64_t cal_n[10] = {}, cal_on[10] = {};
    // depth 3 / per-stage undo (the fourth line)
    int64_t held_n[2] = {}; double held_ms[2] = {0, 0};    // [0] first hit after a fresh window, [1] second and on
    int64_t c_made = 0, c_launched = 0, c_on = 0, c_doomed = 0;
    int64_t undo_jobs = 0, stage_waits = 0; double stage_wait_ms = 0;
    int64_t slot_waits = 0; double slot_wait_ms = 0;
    int64_t restage = 0, restage_spec = 0;
    int64_t chain_waits = 0; double chain_wait_ms = 0;
    int64_t chain_late = 0, chain_steps = 0, chain_launches = 0, chain_cancel = 0;
    int64_t case_hits[16] = {};     // the verdict cases (see PlCase)
};

// The verdict cases as counted (a verdict can count under two: 5/6 also count their 1x/2/3)
enum PlCase : int {
    kCase0 = 0, kCase1a = 1, kCase1b = 2, kCase1c = 3, kCase1d = 4, kCase2 = 5, kCase3 = 6, kCase4 = 7, kCase5 = 8,
    kCase6 = 9, kCase7 = 10, kCase8 = 11, kCase9 = 12, kCase10 = 13, kCase11 = 14, kCaseN = 15
};
const char* case_name(int c);

// What CHECK_STATE compares at the end of a request whose committed tokens end at `end` (section "阻断 3"); the
// symbolic test and the engine's state dump use this one definition
struct StateMask {
    bool gdn = true;               // every front stage's GDN recurrent and conv state (gdn_snapshot_bytes): whole
    int64_t kv_cells = 0;          // QSA KV cells [0, kv_cells)
    int64_t idx_blocks = 0;        // indexer pooled blocks [0, idx_blocks) (n_complete)
    int idx_tail = 0;              // indexer tail slots [0, idx_tail)
    bool ple = true;               // ss.ple.hist (device, stage 0) and ss.ple_prev (host)
    bool block_pos = false;        // written, never read on the device
    bool drafter = false;          // only the drafts depend on it
};
inline StateMask pl_state_mask(int64_t end) {
    StateMask m;
    m.kv_cells = end;
    m.idx_blocks = end / 4;
    m.idx_tail = (int) (end % 4);
    return m;
}

class Sched {
public:
    // Whether a configuration can run (what begin checks first): the engine asks before it commits a request to the
    // new loop, so a configuration it cannot run falls back to today's loop instead of failing the request
    static bool check(const Config& c, std::string& err);
    // A request: the first window (T = 1, token x at position p), ss.ple_prev now, the loop's clock now
    bool begin(const Config& c, int64_t p, int32_t x, const int32_t ple_prev[2], double t0_ms, std::string& err);
    // One round of the loop: 1 go on, 0 the loop is over (nothing in flight; the engine drains), -1 an error (err())
    int step(PlIo& io);
    const std::string& err() const { return err_; }
    const Stats& stats() const { return st_; }
    const Config& config() const { return c_; }

    // diagnostics (the watchdog line, the symbolic test)
    int nwin() const { return nwin_; }
    const PW& win(int k) const { return win_[k]; }
    int ndoom() const { return ndoom_; }
    const PW& doom(int k) const { return doom_[k]; }
    const UndoJob& undo() const { return undo_; }
    bool ending() const { return ending_; }
    int chain_kind() const { return ch_.kind; }
    bool is_doomed(uint64_t uid) const;
    int slot_owner_count() const;
    bool in_flight() const;        // any window in flight on any stage
    int64_t committed_end() const { return end_; }
    // One line of the scheduler's state (an error's context): the windows, the doomed, the undo, the chain, the slots
    std::string dump() const;

private:
    struct Held { uint64_t uid = 0; int slot = -1; bool need_undo = false, need_chain = false; };
    struct ChainState {
        int kind = 0;              // 0: no chain being read; 1 forced through windows in flight; 2 fresh
        int n = 0;                 // outputs (chain_n)
        int src_seq = 0;
        bool early_used = false;   // kind 2: the fresh window's size decided
        uint64_t a_uid = 0;        // kind 2: the fresh window
        int targets = 0;           // new windows still to make (after the fresh window)
        bool done_first = false;   // the first new window decided (today's b_done)
        uint64_t pred_uid = 0;     // the window the next new one follows
        int base = 0;              // the chain output of the next new window's row 0
        float pa = 1.0f;           // the estimate that every window up to `base` is accepted whole
        bool pa_ready = false;     // pa covers the forced windows / the fresh window (else computed at the decision)
        int n_force = 0;
        int32_t force[16] = {};
        float force_prob[16] = {}; // the forced windows' own probabilities (STRATA_PIPELINE_AGREE=0)
        bool cancelled = false;
        bool open = false;         // chain_stop: steps may still be queued
        int asked = 0;             // chain_stop: outputs asked for (chain_extend)
        int queued = 0;            // outputs queued
    };
    struct ChainReq {
        bool pending = false;
        int kind = 0, src_slot = -1, src_seq = 0, T = 0, a = 0;
        uint64_t src_uid = 0, a_uid = 0;
        int64_t p = 0;
        int32_t outp[kMaxRows] = {};
        double t0 = 0.0;
        bool waited = false;
    };

    bool fail(const std::string& m);
    bool run(PlIo& io, PlOp o);
    bool pump(PlIo& io, PW& w, bool doomed);
    bool undo_round(PlIo& io);
    void release();
    bool chain_round(PlIo& io);
    bool decide_fresh(PlIo& io, PW& A, const int32_t* oc, const float* op, int avail, bool early);
    bool decide_next(PlIo& io, int k, bool done);
    bool pick_lookup(PlIo& io, PW& w, const int32_t* chain0);
    bool launch_chain(PlIo& io, const ChainReq& r);
    bool request_round(PlIo& io);
    bool try_prestage(PlIo& io, int k);
    bool launches(PlIo& io, bool& stop);
    bool last_stage(PlIo& io);
    bool verdict(PlIo& io);
    void drop_from(int k);
    int alloc_slot(PlIo& io, PW& w, bool& waited);
    void free_slot(int q, uint64_t uid);
    int t_rule(const float* pr, int from, int avail) const;
    void ple_adv(int32_t ple[2], const int32_t* tok, int n) const;
    PW* find(uint64_t uid);

    Config c_;
    std::string err_;
    Stats st_;
    PW win_[kMaxWin];
    int nwin_ = 0;
    PW doom_[kMaxWin];
    int ndoom_ = 0;
    UndoJob undo_;
    Held held_[4];
    int nheld_ = 0;
    ChainState ch_;
    ChainReq req_;
    uint64_t owner_[kMaxSlots] = {};
    bool src_armed_[kMaxSlots] = {};
    bool ending_ = false;
    uint64_t next_uid_ = 1;
    int32_t outp_[kMaxRows] = {};
    double last_verdict_ = 0.0;
    bool prev_held_ = false;       // the previous verdict's window was itself a held guess (H1 vs H2)
    int64_t end_ = 0;              // the committed tokens end here
    int32_t undo_ple_[2] = {-1, -1};   // N2: today's undo_ple (checked against the verified window's ple_in)
    double slot_wait_t0_ = -1.0;
};

}  // namespace strata::pl
