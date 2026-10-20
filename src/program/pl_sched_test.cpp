// pl_sched_test: the pipelined decode scheduler (pl_sched.hpp) against a symbolic GPU.  CPU only.
//
// The symbolic GPU (Sim) runs the scheduler's operations the way the engine's streams would: one FIFO stream per
// stage, a side stream per front stage (the GDN snapshots), the drafter's stream, events, and the zero-doorbell
// graphs' host flag (a window's graph starts only once the host has serviced it).  Every stage's state is symbolic:
//   - the GDN/conv state as (length, hash of the tokens committed) - a commit replays its window's inputs only onto
//     the state that window read, anything else is poison;
//   - the KV cells, the indexer's tail and pooled blocks, per position (right / wrong);
//   - stage 0's PLE: ss.ple_prev (host) and the device history; every verifier's host staging (last_*), arena
//     (replay inputs), tail and history snapshots, hand-offs, final rows, the GDN snapshots - each tagged with the
//     window that wrote it.
// A window reads them when its graph starts; a window that read anything stale computes garbage, and a verified
// window must have read only the serial loop's state (checked at its verdict, and its picks are then compared with
// the serial loop's tokens).  Hazards checked directly: a commit or restore of another window's data, a torn
// snapshot restored, a hand-off overwritten while its reader still has to read it, the chain's source rows
// overwritten before its copy, a verifier launched or staged while in flight, a host spin that can never end.
//
// Inputs: (a) a recorded 4-stage decode trace's verdicts (embedded below), (b) random worlds (1e6 verdicts per
// mode), (c) exhaustive small worlds with every ending (max_new, EOS, cancel, context edge) at every verdict, (d) the
// old loop (generate.cpp's pw-2 loop, transliterated below as OldLoop) against N2: the same world and timing must
// give the same operations, events and host times, line for line.
//
// usage: pl_sched_test [--quick] [--random N] [--only a|b|c|d] [--seed S] [--dump DIR]
#include "pl_sched.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace strata::pl;

namespace {

constexpr float kHigh = 0.95f, kMid = 0.5f, kLow = 0.05f;
constexpr int32_t kPoison = -7;
constexpr int32_t kEos = 5;
constexpr double kInf = std::numeric_limits<double>::infinity();

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(mix64(seed + 0x1234567ull)) {}
    uint64_t next() { return s = mix64(s); }
    double u() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }
    int i(int n) { return (int) (next() % (uint64_t) n); }
    bool p(double x) { return u() < x; }
};

// ---------------------------------------------------------------------------------------------------------------
// The world: the serial loop's token at every position, and what the drafter guesses there (with a right context:
// every token before it right; with a wrong one), with which probability.  Positions < p0 are the prompt.
struct World {
    int64_t p0 = 64;
    uint64_t salt = 1;
    std::vector<uint8_t> ok, okw, lk, lkok, lk_t;
    std::vector<float> pr, prw;
    std::vector<uint8_t> set_c, set_w;   // trace input: entries set by a verdict record (first writer wins)
    double def_ok = 0.6;
    int64_t eos_pos = -1;
    int64_t max_new = 100;
    int64_t cancel_at = -1;              // the verdict (0-based) after which the request is cancelled
    int64_t max_context = (int64_t) 1 << 40;
    int lookup_t = 0;                    // > 0: lookups offered where lk is set, the policy takes up to this many rows

    int32_t truth(int64_t q) const {
        if (q == eos_pos) return kEos;
        return 1000 + (int32_t) (mix64((uint64_t) q * 0x9e37u ^ salt) % 100000);
    }
    bool in(int64_t q) const { return q >= p0 && q < p0 + (int64_t) ok.size(); }
    size_t ix(int64_t q) const { return (size_t) (q - p0); }
    void size(int64_t n) {
        ok.assign((size_t) n, 1);
        okw.assign((size_t) n, 0);
        lk.assign((size_t) n, 0);
        lkok.assign((size_t) n, 1);
        lk_t.assign((size_t) n, 0);
        pr.assign((size_t) n, kHigh);
        prw.assign((size_t) n, kHigh);
        set_c.assign((size_t) n, 0);
        set_w.assign((size_t) n, 0);
    }
    bool g_ok(int64_t q) const {
        if (in(q)) return ok[ix(q)] != 0;
        return (double) (mix64((uint64_t) q * 7 + salt) % 1000) < def_ok * 1000.0;
    }
    float g_pr(int64_t q) const {
        if (in(q)) return pr[ix(q)];
        const uint64_t h = mix64((uint64_t) q * 13 + salt) % 10;
        return h < 7 ? kHigh : h < 9 ? kMid : kLow;
    }
    bool g_okw(int64_t q) const { return in(q) ? okw[ix(q)] != 0 : false; }
    float g_prw(int64_t q) const {
        if (in(q)) return prw[ix(q)];
        const uint64_t h = mix64((uint64_t) q * 17 + salt) % 10;
        return h < 5 ? kHigh : h < 7 ? kMid : kLow;
    }
    bool g_lk(int64_t q) const { return in(q) && lk[ix(q)] != 0; }
    bool g_lkok(int64_t q) const { return in(q) ? lkok[ix(q)] != 0 : true; }
    int g_lk_t(int64_t q) const { return in(q) && lk_t[ix(q)] != 0 ? lk_t[ix(q)] : lookup_t; }
};

// the GDN/conv state of one stage: the tokens committed, as (length, hash of the tokens from p0 on)
struct GState {
    int64_t len = 0;
    uint64_t h = 0;
    bool poison = false;
    bool operator==(const GState& o) const { return len == o.len && h == o.h && poison == o.poison; }
    bool operator!=(const GState& o) const { return !(*this == o); }
};
uint64_t hstep(uint64_t h, int32_t t) { return mix64(h ^ (uint64_t) (uint32_t) t) + 0x51ed27u; }

// per position right/wrong from a base on (below it: the prompt, right); the first wrong position is kept in a set
struct CellMap {
    int64_t base = 0, front = 0;   // positions >= front were never written
    std::vector<uint8_t> v;         // [pos - base]: 1 right, 2 wrong
    std::set<int64_t> bad;          // wrong positions in [base, front)
    void reset(int64_t b) {
        base = b;
        front = b;
        v.clear();
        bad.clear();
    }
    void set(int64_t x, bool good) {
        if (x < base) return;   // the prompt: never rewritten by a window (a tail restore writes back what it was)
        if (x >= front) {
            for (int64_t y = front; y < x; ++y) bad.insert(y);   // a gap: never written
            if ((size_t) (x - base + 1) > v.size()) v.resize((size_t) (x - base + 1) + 256, 0);
            for (int64_t y = front; y < x; ++y) v[(size_t) (y - base)] = 2;
            front = x + 1;
        }
        v[(size_t) (x - base)] = good ? 1 : 2;
        if (good) bad.erase(x);
        else bad.insert(x);
    }
    bool at(int64_t x) const { return x < base ? true : x >= front ? false : v[(size_t) (x - base)] == 1; }
    bool clean_below(int64_t p) const { return p <= front && (bad.empty() || *bad.begin() >= p); }
};

struct Timing {
    uint64_t seed = 1;
    double jitter = 0.05;
    double stage_scale = 1.0;
    double chain_step = 1.56, chain_round = 1.9;
    double host_iter = 0.002;
    bool clamp = true;   // chain_launch clamps n_out at the context's end (the old drafter did not)
    // stress: a GDN snapshot copy that crawls (its event is what orders the next state write), and a chain whose
    // source copy crawls (source_done is what orders the slot's next last-stage window: B0/B only, the old loop has
    // no such wait and relies on the copy taking microseconds)
    double snap_slow = 0.0, src_slow = 0.0;
    // chain_n() as MtpDrafter::chain_n reports it: the outputs queued so far (with n_queue, the steps queued up to now).
    // false: the clamped n_out while the chain runs (what this mock answered before; MtpDrafter answers that through
    // chain_cap) - both must work, since the scheduler reads the bound from chain_cap and chain_n only once it is done
    bool chain_n_engine = true;
};

// durations (ms) of a window's stage by its rows, from the recorded trace
const double kFront[9] = {0, 7.11, 9.25, 11.95, 14.15, 18.4, 20.9, 23.8, 26.7};
const double kLast[9] = {0, 8.57, 10.6, 13.45, 16.1, 21.6, 24.6, 27.6, 30.6};
const double kStageOff[4] = {0.10, -0.09, -0.02, 0.0};

std::string fmt_op(const PlOp& o) {
    char b[512];
    int n = std::snprintf(b, sizeof b, "%s st=%d slot=%d seq=%d uid=%llu T=%d keep=%d p=%lld spec=%d", op_name(o.op), o.st,
                          o.slot, o.seq, (unsigned long long) o.uid, o.T, o.keep, (long long) o.p, o.spec ? 1 : 0);
    if (o.op == Op::Prestage || o.op == Op::Launch)
        for (int i = 0; i < o.T && n < 480; ++i) n += std::snprintf(b + n, sizeof b - (size_t) n, " %d", o.tok[i]);
    if (o.op == Op::Prestage || o.op == Op::PleSet)
        n += std::snprintf(b + n, sizeof b - (size_t) n, " ple=%d,%d", o.ple[0], o.ple[1]);
    if (o.op == Op::ChainLaunch) {
        n += std::snprintf(b + n, sizeof b - (size_t) n, " kind=%d a=%d nf=%d nout=%d nearly=%d nq=%d f=", o.kind, o.a,
                           o.n_force, o.n_out, o.n_early, o.n_queue);
        for (int i = 0; i < o.n_force && n < 480; ++i) n += std::snprintf(b + n, sizeof b - (size_t) n, "%d,", o.force[i]);
        for (int i = 0; i < o.T && n < 480; ++i) n += std::snprintf(b + n, sizeof b - (size_t) n, " o%d", o.outp[i]);
    }
    if (o.op == Op::ChainExtend) n += std::snprintf(b + n, sizeof b - (size_t) n, " k=%d", o.k);
    return b;
}

// ---------------------------------------------------------------------------------------------------------------
// The symbolic GPU and host (the engine's side of PlIo)
class Sim final : public PlIo {
public:
    std::string err;                 // the first hard error
    std::vector<std::string> log;    // operations and events (with host times), when keep_log
    bool keep_log = false;
    uint64_t progress = 0;           // anything happened this iteration (the harness skips idle time otherwise)
    double t = 0.0;                  // host time (ms)
    std::function<bool(uint64_t)> ever_doomed;
    // results
    std::vector<int32_t> emitted;
    int64_t emit_end = 0;            // the position after the last emitted token
    int64_t verdicts = 0, produced = 0;
    bool eos = false;
    int64_t clamps = 0, hazards_doomed = 0, restage_launch = 0;
    int64_t hand_overwrite_unread = 0;
    int64_t last_end = 0;            // the committed tokens end here (the last verdict's p + a + 1)

    Sim(const World& w, const Config& c, const Timing& tm) : W_(w), C_(c), T_(tm), rng_(tm.seed) {
        nst_ = c.n_front + 1;
        n_streams_ = nst_ + c.n_front + 1;
        q_.assign((size_t) n_streams_, {});
        run_.assign((size_t) n_streams_, -1);
        free_.assign((size_t) n_streams_, 0.0);
        st_.assign((size_t) nst_, Stage{});
        const int64_t b0 = (W_.p0 - 8) & ~(int64_t) 3;   // the prompt's last cells and blocks: right
        for (Stage& s : st_) {
            s.state = GState{W_.p0, 0, false};
            s.kv.reset(b0);
            s.idx.reset(b0);
            s.pooled.reset(b0 / 4);
            for (int64_t x = b0; x < W_.p0; ++x) {
                s.kv.set(x, true);
                s.idx.set(x, true);
            }
            for (int64_t b = b0 / 4; b < W_.p0 / 4; ++b) s.pooled.set(b, true);
        }
        ple_[0] = W_.truth(W_.p0 - 2);
        ple_[1] = W_.truth(W_.p0 - 1);
        hist_pos_ = W_.p0 - 1;
        hist_ok_ = true;
        emit_end = W_.p0 + 1;   // the prefill emitted the token at p0
        ht_.assign(1, 0);
        ver_.assign((size_t) (nst_ * kMaxSlots), Verif{});
        for (auto& b : hand_) for (auto& h : b) h = Hand{};
    }

    // ---- PlIo
    int front_poll(int st, int slot, bool doomed) override { (void) doomed; return poll(st, slot, nullptr); }
    int last_poll(int slot, int32_t* outp) override { return poll(C_.n_front, slot, outp); }
    int chain_ready() override {
        t += 0.0005;
        advance(t);
        if (!ch_.live) return note_ready(ch_.n);
        int k = 0;
        while (k < ch_.n_early && k < ch_.queued && ch_.landed[k]) ++k;
        for (int j = 0; j < k; ++j) {
            tok_[j] = ch_.out[j];
            prob_[j] = ch_.prob[j];
        }
        return note_ready(k);
    }
    int chain_poll() override {
        t += 0.0005;
        advance(t);
        if (!ch_.live) return 1;
        if (ch_.open) return 0;
        if (ch_.queued == 0 || !ch_.landed[ch_.queued - 1]) return 0;
        ch_.live = false;
        ch_.n = ch_.queued;
        for (int j = 0; j < 12; ++j) {
            tok_[j] = j < ch_.n ? ch_.out[j] : 0;
            prob_[j] = j < ch_.n ? ch_.prob[j] : 0.0f;
        }
        ++progress;
        return 1;
    }
    bool chain_live() override { return ch_.live; }
    int chain_cap() override { return ch_.cap; }
    int chain_n() override { return T_.chain_n_engine && ch_.live ? ch_.queued : ch_.n; }
    const int32_t* chain_tok() override { return tok_; }
    const float* chain_prob() override { return prob_; }
    void ple_prev(int32_t out[2]) override {
        out[0] = ple_[0];
        out[1] = ple_[1];
    }
    int lookup_propose(int max_k, Lookup& out) override {
        out = Lookup{};
        if (W_.lookup_t <= 0) return 0;
        const int64_t q0 = emit_end;   // the window's row 0 sits at q0 - 1
        if (!W_.g_lk(q0 - 1)) return 0;
        const int k = std::min(max_k, 15);
        for (int i = 0; i < k; ++i) out.tok[i] = W_.g_lkok(q0 + i) ? W_.truth(q0 + i) : W_.truth(q0 + i) + 1;
        out.k = k;
        out.match = 8;
        return k;
    }
    bool lookup_choose(int T, int k, int match, int& tt) override {
        (void) T;
        (void) match;
        tt = std::min(W_.g_lk_t(emit_end - 1), k + 1);
        return tt >= 2;
    }
    double now_ms() override { return t; }
    void trace(const char* ev, int seq, int x0, int x1) override {
        if (keep_log) {
            char b[96];
            std::snprintf(b, sizeof b, "%.6f %s %d %d %d", t, ev, seq, x0, x1);
            log.emplace_back(b);
        }
        events.push_back(Ev{ev, seq, x0, x1});
    }
    void observe(const PW& w, int a, double ms) override {
        (void) w;
        (void) a;
        (void) ms;
    }
    bool verdict(const PW& w, int a, const int32_t* outp, bool& last) override {
        t += 0.01;
        ++progress;
        // the verified window read the serial loop's state on every stage
        auto it = std::find_if(win_rec_.begin(), win_rec_.end(), [&](const WinRec& r) { return r.uid == w.uid; });
        if (it == win_rec_.end()) return hard("a verified window that never ran");
        for (int s = 0; s < nst_; ++s) {
            if (!(it->ran & (1u << s))) return hard("a verified window that skipped stage " + std::to_string(s));
            if (!(it->valid & (1u << s)))
                return hard("a verified window read a stale state on stage " + std::to_string(s) + " (" + it->why[s] + ")");
        }
        if (w.p != emit_end - 1) return hard("a verified window that is not at the emitted text's end");
        for (int s = 0; s < nst_; ++s) {   // each stage ran the verified windows in their order
            if (it->order[s] <= verified_order_[s])
                return hard("stage " + std::to_string(s) + " ran a verified window before the one verified before it");
            verified_order_[s] = it->order[s];
        }
        for (int i = 0; i <= a && !eos && produced < W_.max_new; ++i) {
            const int64_t pos = w.p + 1 + i;
            if (outp[i] != W_.truth(pos))
                return hard("an emitted token differs from the serial loop's at " + std::to_string(pos));
            emitted.push_back(outp[i]);
            ++produced;
            emit_end = pos + 1;
            if (outp[i] == kEos) eos = true;
        }
        last = eos || produced >= W_.max_new || (W_.cancel_at >= 0 && verdicts >= W_.cancel_at);
        ++verdicts;
        last_end = w.p + a + 1;
        return true;
    }
    bool exec(const PlOp& o) override;

    // ---- the harness
    struct Ev {
        std::string ev;
        int seq, x0, x1;
    };
    std::vector<Ev> events;
    // the next time anything on the GPU can happen (kInf: nothing will)
    double next_gpu_time() {
        double best = kInf;
        for (int s = 0; s < n_streams_; ++s) {
            if (run_[(size_t) s] >= 0) best = std::min(best, ops_[(size_t) run_[(size_t) s]].te);
            else if (!q_[(size_t) s].empty()) best = std::min(best, start_time(q_[(size_t) s].front(), s));
        }
        return best;
    }
    void advance(double until) { run_until(until); }
    // after the loop: today's pl_drain (every window in flight served to its end) and the drafter's chain drained
    bool drain(std::string& why) {
        for (Verif& v : ver_)
            if (v.fl) {
                why = "a window still in flight after the loop";
                return false;
            }
        run_until(kInf);
        for (int s = 0; s < n_streams_; ++s)
            if (!q_[(size_t) s].empty() || run_[(size_t) s] >= 0) {
                why = "a stream never drained (it waits for something nobody does)";
                return false;
            }
        ch_.live = false;
        return true;
    }
    // CHECK_STATE: every stage against the serial loop's state at the committed end, by pl_state_mask
    bool check_state(int64_t end, std::string& why) {
        const StateMask m = pl_state_mask(end);
        const GState ref = truth_state(end);
        for (int s = 0; s < nst_; ++s) {
            Stage& S = st_[(size_t) s];
            if (m.gdn && S.state != ref) { why = "GDN state of stage " + std::to_string(s); return false; }
            if (!S.kv.clean_below(m.kv_cells)) { why = "KV cells of stage " + std::to_string(s); return false; }
            if (!S.pooled.clean_below(m.idx_blocks)) { why = "indexer pooled of stage " + std::to_string(s); return false; }
            for (int i = 0; i < m.idx_tail; ++i)
                if (!S.idx.at(4 * m.idx_blocks + i)) { why = "indexer tail of stage " + std::to_string(s); return false; }
        }
        if (m.ple) {
            if (ple_[0] != W_.truth(end - 2) || ple_[1] != W_.truth(end - 1)) { why = "ss.ple_prev"; return false; }
            if (hist_pos_ != end - 1 || !hist_ok_) { why = "the PLE history"; return false; }
        }
        return true;
    }
    int64_t committed_tokens() const { return st_[0].state.len; }

private:
    struct Hand { uint64_t uid = 0; bool valid = false, read = true; };
    struct Snap { uint64_t uid = 0; GState v; bool torn = false; };
    struct Arena { uint64_t uid = 0; GState read; bool valid = false; int T = 0; int64_t p = 0; int32_t tok[8] = {}; bool rows[8] = {}; };
    struct TailSnap { uint64_t uid = 0; int64_t p = 0; bool ok[4] = {}; };
    struct HistSnap { uint64_t uid = 0; int64_t p = 0; bool rows[8] = {}; };
    struct Stage {
        GState state;
        uint64_t ver = 0;          // bumped by every write of the state
        CellMap kv, idx, pooled;
        Snap buf[kMaxSlots];
        Arena arena[kMaxSlots];
        TailSnap tail[kMaxSlots];
        int copying = -1;          // the snapshot copy running on its side stream (op index)
        int64_t pre_inst[kMaxSlots] = {-1, -1, -1, -1, -1}, ev_inst[kMaxSlots] = {-1, -1, -1, -1, -1};
    };
    struct Verif {
        bool fl = false, flag = false, prestaged = false, commit_live = false;
        double flag_t = 0.0;
        int graph = -1, commit = -1;
        int last_t = 0;
        int64_t last_pos0 = 0;
        int32_t last_tok[8] = {};
        uint64_t owner = 0;
        int32_t pl_prev[2] = {-1, -1};
    };
    enum GK : uint8_t { G_WIN, G_COMMIT, G_RESTORE, G_SNAP, G_REC, G_WAIT, G_SRC, G_STEP };
    struct GOp {
        GK k;
        int stream = 0;
        double dur = 0.0, t_enq = 0.0, ts = 0.0, te = 0.0;
        bool started = false, done = false;
        int st = 0, slot = 0, T = 0, keep = 0, verif = -1, out_j = -1;
        uint64_t uid = 0;
        int64_t p = 0, inst = -1;
        int32_t tok[8] = {};
        int32_t ple[2] = {-1, -1};
        // results
        bool valid = false;
        bool rows[8] = {};
        int32_t outp[8] = {};
        GState read;
        uint64_t ver0 = 0;
        bool torn = false;
        std::string why;
    };
    struct WinRec {
        uint64_t uid;
        uint32_t ran = 0, valid = 0;
        int64_t order[kMaxFront + 1] = {};   // its launch's place in each stage's stream
        std::string why[kMaxFront + 1];
    };
    struct Chain {
        bool live = false, open = false;
        int n = 0, n_early = 0, queued = 0;
        int cap = 0;                 // n_out after the clamp (n becomes the outputs queued once the chain is done)
        int32_t out[12] = {};
        float prob[12] = {};
        bool landed[12] = {};
    };

    bool hard(const std::string& m) {
        if (err.empty()) err = m;
        return false;
    }
    int note_ready(int k) {
        if (k != last_ready_) {
            last_ready_ = k;
            ++progress;
        }
        return k;
    }
    double jit(double d) { return d * T_.stage_scale * (1.0 + T_.jitter * (2.0 * rng_.u() - 1.0)); }
    GState truth_state(int64_t len) {
        if (len < W_.p0) return GState{len, 0, true};
        while ((int64_t) ht_.size() <= len - W_.p0)
            ht_.push_back(hstep(ht_.back(), W_.truth(W_.p0 + (int64_t) ht_.size() - 1)));
        return GState{len, ht_[(size_t) (len - W_.p0)], false};
    }
    Verif& V(int st, int slot) { return ver_[(size_t) (st * kMaxSlots + slot)]; }
    WinRec& rec(uint64_t uid) {
        for (auto it = win_rec_.rbegin(); it != win_rec_.rend(); ++it)
            if (it->uid == uid) return *it;
        win_rec_.push_back(WinRec{uid});
        if (win_rec_.size() > 64) win_rec_.erase(win_rec_.begin());
        return win_rec_.back();
    }
    int push(GOp o, int stream) {
        o.stream = stream;
        o.t_enq = t;
        ops_.push_back(o);
        const int id = (int) ops_.size() - 1;
        q_[(size_t) stream].push_back(id);
        return id;
    }
    int64_t new_inst() {
        inst_t_.push_back(-1.0);
        inst_ver_.push_back(0);
        return (int64_t) inst_t_.size() - 1;
    }
    int side(int st) const { return nst_ + st; }
    int drafter() const { return nst_ + C_.n_front; }
    double start_time(int id, int s) const {
        const GOp& o = ops_[(size_t) id];
        double b = std::max(free_[(size_t) s], o.t_enq);
        if (o.k == G_WAIT) {
            if (o.inst < 0) return b;
            const double it = inst_t_[(size_t) o.inst];
            if (it < 0.0) return kInf;
            return std::max(b, it);
        }
        if (o.k == G_WIN) {
            const Verif& v = ver_[(size_t) o.verif];
            if (!v.flag || v.graph != id) return kInf;
            return std::max(b, v.flag_t);
        }
        return b;
    }
    void run_until(double until) {
        for (;;) {
            double best = kInf;
            int bs = -1;
            bool is_end = false;
            for (int s = 0; s < n_streams_; ++s) {
                if (run_[(size_t) s] >= 0) {
                    const double te = ops_[(size_t) run_[(size_t) s]].te;
                    if (te < best || (te == best && !is_end)) {
                        best = te;
                        bs = s;
                        is_end = true;
                    }
                } else if (!q_[(size_t) s].empty()) {
                    const double ts = start_time(q_[(size_t) s].front(), s);
                    if (ts < best) {
                        best = ts;
                        bs = s;
                        is_end = false;
                    }
                }
            }
            if (bs < 0 || best > until) return;
            gt_ = best;
            if (is_end) {
                const int id = run_[(size_t) bs];
                finish(ops_[(size_t) id]);
                ops_[(size_t) id].done = true;
                run_[(size_t) bs] = -1;
                free_[(size_t) bs] = best;
                q_[(size_t) bs].pop_front();
            } else {
                const int id = q_[(size_t) bs].front();
                GOp& o = ops_[(size_t) id];
                o.started = true;
                o.ts = best;
                o.te = best + o.dur;
                start(o, id);
                run_[(size_t) bs] = id;
            }
        }
    }
    void state_write(int st) {
        Stage& S = st_[(size_t) st];
        ++S.ver;
        if (S.copying >= 0) ops_[(size_t) S.copying].torn = true;
    }
    void start(GOp& o, int id);
    void finish(GOp& o);
    int poll(int st, int slot, int32_t* outp);
    bool spin_until_done(int id, const char* what) {
        for (int guard = 0; guard < 1000000; ++guard) {
            advance(t);
            if (ops_[(size_t) id].done) return true;
            const double nx = next_gpu_time();
            if (nx == kInf) return hard(std::string("deadlock: the host spins on ") + what + " that can never finish");
            t = std::max(t, nx);
        }
        return hard("spin guard");
    }
    void compute_chain(const PlOp& o);

    const World& W_;
    Config C_;
    Timing T_;
    Rng rng_;
    int nst_ = 0, n_streams_ = 0;
    std::vector<GOp> ops_;
    std::vector<std::deque<int>> q_;
    std::vector<int> run_;
    std::vector<double> free_;
    std::vector<double> inst_t_;
    std::vector<uint64_t> inst_ver_;
    std::vector<Stage> st_;
    std::vector<Verif> ver_;
    Hand hand_[kMaxFront][kMaxSlots];
    struct { uint64_t uid = 0; bool valid = false; } final_[kMaxSlots];
    HistSnap hist_snap_[kMaxSlots];
    int64_t src_inst_[kMaxSlots] = {-1, -1, -1, -1, -1};
    int32_t ple_[2] = {-1, -1};
    int64_t hist_pos_ = 0;
    bool hist_ok_ = false;
    std::vector<uint64_t> ht_;
    std::vector<WinRec> win_rec_;
    Chain ch_;
    int32_t tok_[12] = {};
    float prob_[12] = {};
    int last_ready_ = -1;
    double gt_ = 0.0;
    int64_t launches_[kMaxFront + 1] = {};       // windows started on each stage so far
    int64_t verified_order_[kMaxFront + 1] = {}; // the last verified window's place on each stage
    uint64_t chain_src_uid_ = 0;
    int chain_src_slot_ = -1;

    friend struct Harness;
};

int Sim::poll(int st, int slot, int32_t* outp) {
    t += 0.0005;
    if (slot < 0 || slot >= kMaxSlots) return hard("a poll of no slot"), -1;
    Verif& v = V(st, slot);
    if (!v.fl) return hard("a poll of a verifier with nothing in flight"), -1;
    if (!v.flag) {   // service(): the zero-doorbell graph's flag 1
        v.flag = true;
        v.flag_t = t;
        ++progress;
    }
    advance(t);
    const GOp& g = ops_[(size_t) v.graph];
    if (!g.done) return 0;
    v.fl = false;
    if (outp != nullptr)
        for (int i = 0; i < g.T; ++i) outp[i] = g.outp[i];
    ++progress;
    return 1;
}

void Sim::start(GOp& o, int id) {
    switch (o.k) {
        case G_WIN: {
            Stage& S = st_[(size_t) o.st];
            bool ok = true;
            std::string why;
            if (S.state != truth_state(o.p)) { ok = false; why = "GDN"; }
            if (ok && !S.kv.clean_below(o.p)) { ok = false; why = "KV"; }
            if (ok && !S.pooled.clean_below(o.p / 4)) { ok = false; why = "pooled"; }
            TailSnap& ts = S.tail[o.slot];
            ts = TailSnap{o.uid, o.p, {}};
            for (int64_t x = 4 * (o.p / 4); x < o.p; ++x) {
                ts.ok[x - 4 * (o.p / 4)] = S.idx.at(x);
                if (ok && !S.idx.at(x)) { ok = false; why = "tail"; }
            }
            if (o.st == 0) {
                if (ok && (o.ple[0] != W_.truth(o.p - 2) || o.ple[1] != W_.truth(o.p - 1))) { ok = false; why = "PLE prev"; }
            } else {
                Hand& h = hand_[o.st - 1][o.slot];
                if (h.uid != o.uid) {
                    ok = false;
                    why = "hand-off of another window";
                } else {
                    h.read = true;
                    if (ok && !h.valid) { ok = false; why = "hand-off"; }
                }
            }
            o.read = S.state;
            o.valid = ok;
            o.why = why;
            WinRec& r = rec(o.uid);
            r.ran |= 1u << o.st;
            r.order[o.st] = ++launches_[o.st];
            if (ok) r.valid |= 1u << o.st;
            else r.why[o.st] = why;
            break;
        }
        case G_COMMIT: {
            Stage& S = st_[(size_t) o.st];
            Arena& A = S.arena[o.slot];
            if (A.uid != o.uid) { hard("a commit replays another window's inputs (stage " + std::to_string(o.st) + ")"); break; }
            if (o.keep > A.T) { hard("a commit keeps more rows than its window had"); break; }
            GState nv;
            if (S.state == A.read && A.valid && !S.state.poison) {
                nv = A.read;
                for (int i = 0; i < o.keep; ++i) {
                    nv.h = hstep(nv.h, A.tok[i]);
                    ++nv.len;
                }
            } else {
                nv = GState{S.state.len + o.keep, mix64(S.state.h + 77), true};
            }
            S.state = nv;
            state_write(o.st);
            TailSnap& ts = S.tail[o.slot];
            if (ts.uid != o.uid) { hard("a commit restores another window's indexer tail"); break; }
            for (int64_t x = 4 * (A.p / 4); x < A.p; ++x) S.idx.set(x, ts.ok[x - 4 * (A.p / 4)]);
            for (int i = 0; i < o.keep; ++i) {
                const int64_t x = A.p + i;
                S.idx.set(x, A.rows[i]);
                if (x % 4 == 3) S.pooled.set(x / 4, S.idx.at(x - 3) && S.idx.at(x - 2) && S.idx.at(x - 1) && S.idx.at(x));
            }
            if (o.st == 0) {
                HistSnap& hs = hist_snap_[o.slot];
                if (hs.uid != o.uid) { hard("a commit writes another window's PLE history"); break; }
                hist_pos_ = A.p + o.keep - 1;
                hist_ok_ = hs.rows[o.keep - 1];
            }
            break;
        }
        case G_RESTORE: {
            Stage& S = st_[(size_t) o.st];
            const Snap& b = S.buf[o.slot];
            if (b.uid != o.uid) { hard("an undo restores another window's snapshot"); break; }
            if (b.torn) { hard("an undo restores a torn snapshot"); break; }
            S.state = b.v;
            state_write(o.st);
            break;
        }
        case G_SNAP: {
            Stage& S = st_[(size_t) o.st];
            o.read = S.state;
            o.ver0 = S.ver;
            o.torn = o.inst >= 0 && inst_ver_[(size_t) o.inst] != S.ver;   // the state moved since its pre event
            S.copying = id;
            break;
        }
        case G_REC: {
            inst_t_[(size_t) o.inst] = o.ts;
            if (o.st >= 0) inst_ver_[(size_t) o.inst] = st_[(size_t) o.st].ver;
            break;
        }
        case G_SRC: {
            if (final_[o.slot].uid != o.uid) hard("the chain read another window's final rows (source_done)");
            break;
        }
        default: break;
    }
}

void Sim::finish(GOp& o) {
    switch (o.k) {
        case G_WIN: {
            Stage& S = st_[(size_t) o.st];
            for (int i = 0; i < o.T; ++i) {
                bool r = o.valid;
                for (int j = 0; j <= i && r; ++j) r = o.tok[j] == W_.truth(o.p + j);
                o.rows[i] = r;
            }
            for (int i = 0; i < o.T; ++i) {
                const int64_t x = o.p + i;
                S.kv.set(x, o.rows[i]);
                S.idx.set(x, o.rows[i]);
                if (x % 4 == 3) S.pooled.set(x / 4, S.idx.at(x - 3) && S.idx.at(x - 2) && S.idx.at(x - 1) && S.idx.at(x));
            }
            Arena& A = S.arena[o.slot];
            A.uid = o.uid;
            A.read = o.read;
            A.valid = o.valid;
            A.T = o.T;
            A.p = o.p;
            for (int i = 0; i < o.T; ++i) {
                A.tok[i] = o.tok[i];
                A.rows[i] = o.rows[i];
            }
            if (o.st == 0) {
                HistSnap& hs = hist_snap_[o.slot];
                hs.uid = o.uid;
                hs.p = o.p;
                for (int i = 0; i < o.T; ++i) hs.rows[i] = o.rows[i];
            }
            if (o.st < C_.n_front) {
                Hand& h = hand_[o.st][o.slot];
                if (h.uid != 0 && h.uid != o.uid && !h.read) {
                    if (ever_doomed && ever_doomed(h.uid)) ++hazards_doomed;
                    else {
                        ++hand_overwrite_unread;
                        hard("a hand-off written while its reader still has to read it (stage " + std::to_string(o.st) + ")");
                    }
                }
                h.uid = o.uid;
                h.valid = o.valid;
                h.read = false;
            } else {
                for (int i = 0; i < o.T; ++i) o.outp[i] = o.rows[i] ? W_.truth(o.p + i + 1) : kPoison;
                final_[o.slot].uid = o.uid;
                final_[o.slot].valid = o.valid;
            }
            break;
        }
        case G_SNAP: {
            Stage& S = st_[(size_t) o.st];
            if (S.ver != o.ver0) o.torn = true;
            if (S.copying == (int) (&o - ops_.data())) S.copying = -1;
            S.buf[o.slot] = Snap{o.uid, o.read, o.torn};
            break;
        }
        case G_STEP: {
            ch_.landed[o.out_j] = true;
            break;
        }
        default: break;
    }
}

// the drafter's outputs: output j guesses position p + a + 2 + j from the tokens before it (the verified window's
// bonus at p + a + 1, then the forced tokens, then its own picks); right only with a right context
void Sim::compute_chain(const PlOp& o) {
    const int64_t base = o.p + o.a + 1;
    bool ctx = o.outp[o.a] == W_.truth(base);
    for (int j = 0; j < ch_.n; ++j) {
        const int64_t q = base + 1 + j;
        int32_t g;
        float pr;
        if (ctx) {
            g = W_.g_ok(q) ? W_.truth(q) : W_.truth(q) + 1;
            pr = W_.g_pr(q);
        } else {
            g = W_.g_okw(q) ? W_.truth(q) : W_.truth(q) + 2;
            pr = W_.g_prw(q);
        }
        ch_.out[j] = g;
        ch_.prob[j] = pr;
        const int32_t in = j < o.n_force ? o.force[j] : g;   // the next step's input at q
        ctx = ctx && in == W_.truth(q);
    }
}

bool Sim::exec(const PlOp& o) {
    ++progress;
    if (keep_log) {
        char b[24];
        std::snprintf(b, sizeof b, "%.6f ", t);
        log.push_back(b + fmt_op(o));
    }
    advance(t);
    const int last = C_.n_front;
    switch (o.op) {
        case Op::Gap:
        case Op::AdaptVerdict:
        case Op::AdaptTick:
            t += 0.001;
            return true;
        case Op::Prestage: {
            t += 0.05;
            if (o.slot < 0 || o.slot >= C_.ns()) return hard("prestage: slot out of range");
            Verif& v = V(0, o.slot);
            if (v.fl) return hard("prestage: a window is in flight on this verifier");
            if (o.T < 1 || o.T > C_.S) return hard("prestage: window size out of range");
            v.last_t = o.T;
            v.last_pos0 = o.p;
            for (int i = 0; i < o.T; ++i) v.last_tok[i] = o.tok[i];
            v.owner = o.uid;
            v.pl_prev[0] = o.ple[0];
            v.pl_prev[1] = o.ple[1];
            v.prestaged = true;
            return true;
        }
        case Op::SnapTake: {
            t += 0.005;
            Stage& S = st_[(size_t) o.st];
            GOp r;
            r.k = G_REC;
            r.st = o.st;
            r.inst = new_inst();
            S.pre_inst[o.slot] = r.inst;
            push(r, o.st);
            GOp w;
            w.k = G_WAIT;
            w.inst = r.inst;
            push(w, side(o.st));
            GOp c;
            c.k = G_SNAP;
            c.st = o.st;
            c.slot = o.slot;
            c.uid = o.uid;
            c.inst = r.inst;
            c.dur = rng_.p(T_.snap_slow) ? 15.0 : 0.06;
            push(c, side(o.st));
            GOp e;
            e.k = G_REC;
            e.st = -1;
            e.inst = new_inst();
            S.ev_inst[o.slot] = e.inst;
            push(e, side(o.st));
            return true;
        }
        case Op::SnapWait: {
            t += 0.002;
            GOp w;
            w.k = G_WAIT;
            w.inst = st_[(size_t) o.st].ev_inst[o.slot];
            push(w, o.st);
            return true;
        }
        case Op::Launch: {
            t += 0.03;
            if (o.slot < 0 || o.slot >= C_.ns()) return hard("launch: slot out of range");
            if (o.p + o.T > W_.max_context) return hard("launch: the window runs past the context");
            Verif& v = V(o.st, o.slot);
            if (v.fl) return hard("launch: a window is already in flight on this verifier");
            bool staged = v.prestaged && v.last_t == o.T && v.last_pos0 == o.p && v.owner == o.uid;
            for (int i = 0; staged && i < o.T; ++i) staged = v.last_tok[i] == o.tok[i];
            if (staged && o.st == 0) staged = v.pl_prev[0] == ple_[0] && v.pl_prev[1] == ple_[1];
            v.prestaged = false;
            if (!staged) {   // pl_stage now, from ss.ple_prev
                v.last_t = o.T;
                v.last_pos0 = o.p;
                for (int i = 0; i < o.T; ++i) v.last_tok[i] = o.tok[i];
                v.owner = o.uid;
                v.pl_prev[0] = ple_[0];
                v.pl_prev[1] = ple_[1];
                if (o.st == 0) {
                    ++restage_launch;
                    t += 0.05;
                }
            }
            GOp g;
            g.k = G_WIN;
            g.st = o.st;
            g.slot = o.slot;
            g.uid = o.uid;
            g.T = o.T;
            g.p = o.p;
            for (int i = 0; i < o.T; ++i) g.tok[i] = o.tok[i];
            g.ple[0] = v.pl_prev[0];
            g.ple[1] = v.pl_prev[1];
            g.verif = o.st * kMaxSlots + o.slot;
            const double base = o.st == last ? kLast[std::min(o.T, 8)] : kFront[std::min(o.T, 8)];
            g.dur = jit(base + (o.st < 4 ? kStageOff[o.st] : 0.0));
            const int id = push(g, o.st);
            v.fl = true;
            v.flag = false;
            v.graph = id;
            return true;
        }
        case Op::SrcWait: {
            t += 0.002;
            GOp w;
            w.k = G_WAIT;
            w.inst = src_inst_[o.slot];
            push(w, last);
            return true;
        }
        case Op::Commit: {
            t += 0.01;
            if (o.slot < 0 || o.slot >= C_.ns()) return hard("commit: slot out of range");
            Verif& v = V(o.st, o.slot);
            if (v.owner != o.uid) return hard("commit: the verifier's staging is another window's");
            if (o.keep < 1 || o.keep > v.last_t) return hard("commit: count out of range");
            if (v.commit_live && !spin_until_done(v.commit, "the verifier's previous commit")) return false;
            GOp c;
            c.k = G_COMMIT;
            c.st = o.st;
            c.slot = o.slot;
            c.uid = o.uid;
            c.keep = o.keep;
            c.dur = 0.25;
            v.commit = push(c, o.st);
            v.commit_live = true;
            if (o.st == 0)
                for (int i = 0; i < o.keep; ++i) {
                    ple_[0] = ple_[1];
                    ple_[1] = v.last_tok[i];
                }
            return true;
        }
        case Op::Restore: {
            t += 0.005;
            GOp r;
            r.k = G_RESTORE;
            r.st = o.st;
            r.slot = o.slot;
            r.uid = o.uid;
            r.dur = 0.05;
            push(r, o.st);
            return true;
        }
        case Op::PleSet:
            t += 0.001;
            ple_[0] = o.ple[0];
            ple_[1] = o.ple[1];
            return true;
        case Op::ChainWaitIdle: {
            for (int guard = 0; guard < 1000000 && ch_.live; ++guard) {
                if (chain_poll() == 1) break;
                if (ch_.open) return hard("deadlock: the host waits for an open chain");
                const double nx = next_gpu_time();
                if (nx == kInf) return hard("deadlock: the host waits for a chain that never ends");
                t = std::max(t, nx);
            }
            return true;
        }
        case Op::ChainLaunch: {
            t += 0.05;
            if (ch_.live) return hard("chain_launch: a chain is already in flight");
            if (o.n_out < 1 || o.n_out > kChainMax || o.n_force > o.n_out || o.n_force > 16)
                return hard("chain_launch: arguments out of range");
            if (o.slot < 0 || o.slot >= C_.ns()) return hard("chain_launch: slot out of range");
            ch_ = Chain{};
            ch_.live = true;
            int n = o.n_out;
            const int64_t cap = W_.max_context - (o.p + o.a) - 1;   // the chain's cells stay below max_cells - 1
            if (T_.clamp && n > cap) {
                n = (int) std::max<int64_t>(1, cap);
                ++clamps;
            }
            ch_.n = n;
            ch_.cap = n;
            ch_.n_early = std::min(o.n_early, n);
            compute_chain(o);
            chain_src_uid_ = o.uid;
            chain_src_slot_ = o.slot;
            GOp s;
            s.k = G_SRC;
            s.slot = o.slot;
            s.uid = o.uid;
            s.dur = rng_.p(T_.src_slow) ? 70.0 : 0.01;   // longer than a window's way through every stage
            push(s, drafter());
            GOp r;
            r.k = G_REC;
            r.st = -1;
            r.inst = new_inst();
            src_inst_[o.slot] = r.inst;
            push(r, drafter());
            const int q = o.n_queue > 0 && o.n_queue < n ? o.n_queue : n;
            for (int j = 0; j < q; ++j) {
                GOp g;
                g.k = G_STEP;
                g.out_j = j;
                g.dur = j == 0 ? T_.chain_round * T_.stage_scale : T_.chain_step * T_.stage_scale;
                push(g, drafter());
            }
            ch_.queued = q;
            ch_.open = q < n;
            for (int j = 0; j < 12; ++j) {   // nothing landed yet: what a reader sees before is stale
                tok_[j] = -999;
                prob_[j] = -1.0f;
            }
            last_ready_ = -1;
            return true;
        }
        case Op::ChainExtend: {
            t += 0.005;
            if (!ch_.live || !ch_.open) return true;
            const int want = std::min(ch_.n, o.k + 1);
            for (int j = ch_.queued; j < want; ++j) {
                GOp g;
                g.k = G_STEP;
                g.out_j = j;
                g.dur = T_.chain_step * T_.stage_scale;
                push(g, drafter());
            }
            ch_.queued = std::max(ch_.queued, want);
            if (ch_.queued >= ch_.n) ch_.open = false;
            return true;
        }
        case Op::ChainClose:
            t += 0.001;
            ch_.open = false;
            return true;
    }
    return hard("an unknown operation");
}

// ---------------------------------------------------------------------------------------------------------------
// The old loop: generate.cpp's pw-2 decode loop over PlIo, statement for statement.
// What the derivation keeps:
//   each round  pump0(D) if doomed, pump0(A), pump0(B), then A on the last stage served
//               -> F0 / F1 events
//   undo        doomed && !D.fl: every stage in undo_w.com restored, stage 0's ple_prev = undo_ple, the commit
//               replayed -> U
//   ending      if (ending) { if (!doomed) break; continue; }
//   chain       kind 2 and A undecided: A from the outputs landed -> CE; chain_poll; A from all of
//               them if the chain ended first; B once decided with today's prestage
//               (!VF(0,B).in_flight() && !doomed) -> CD; chain_kind = 0 once it ended
//   A launch    ready, not in flight, front stage left, !doomed; the context ends the loop -> L0 st
//   B launch    one stage behind A: snapshot wait, (stage 0: undo_ple), A's speculative commit, B's snapshot, B;
//               below the gate B.ready = false -> L0 10+st
//   last        A.finished && !A.s1 -> L1
//   verdict     wait for the drafter, commit a+1 on the last stage, emit, V; on: the chain forced
//               through B first, then A's commit on the stages B has not reached, A = B; else: the
//               commit a+1 there, B doomed if launched, the fresh chain, A = nA -> CL
// Operations the engine's translation will issue for them are the PlOp's of pl_sched.hpp, slot = seq & 1.
class OldLoop {
public:
    bool begin(const Config& c, int64_t p, int32_t x, double t0) {
        c_ = c;
        A = PW{};
        B = PW{};
        D = PW{};
        undo_w = PW{};
        doomed = ending = early_used = b_done = false;
        chain_kind = chain_n = undo_keep = 0;
        pl_fm = 0;
        next_uid = 1;
        A.T = 1;
        A.p = p;
        A.tok[0] = x;
        A.ready = true;
        A.uid = next_uid++;
        last_verdict = t0;
        return true;
    }
    bool is_doomed(uint64_t uid) const { return doomed && D.uid == uid; }
    uint64_t doomed_uid() const { return doomed ? D.uid : 0; }
    int step(PlIo& io);
    std::string err;

private:
    bool ex(PlIo& io, const PlOp& o) {
        if (!io.exec(o)) {
            err = std::string("old loop: ") + op_name(o.op);
            return false;
        }
        return true;
    }
    static int slot(const PW& w) { return w.seq & 1; }
    bool pump0(PlIo& io, PW& w, bool is_d) {
        if (!w.fl) return true;
        const int r = io.front_poll(w.fr, slot(w), is_d && doomed);
        if (r < 0) return false;
        if (r == 1) {
            w.fl = false;
            if (++w.fr == c_.n_front) w.finished = true;
            io.trace("F0", w.seq, w.fr, w.spec ? 1 : 0);
        }
        return true;
    }
    int t_rule(const float* pr, int from, int avail) const {
        int T = c_.S_mtp;
        if (c_.spec_min_p > 0.0) {
            T = 1;
            while (T < c_.S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) c_.spec_min_p) ++T;
        }
        return std::max(1, std::min(T, 1 + avail - from));
    }
    void make_b(const int32_t* oc, const float* op, int base, int avail, float pa) {
        B = PW{};
        if (base >= avail) return;
        B.seq = A.seq + 1;
        B.p = A.p + A.T;
        B.tok[0] = oc[base];
        B.T = t_rule(op, base + 1, avail);
        for (int i = 1; i < B.T; ++i) {
            B.tok[i] = oc[base + i];
            B.prob[i - 1] = op[base + i];
        }
        B.p_on = pa * op[base];
        if (c_.force_miss > 0 && ++pl_fm % c_.force_miss == 0) B.tok[0] = B.tok[0] == 0 ? 1 : 0;
        B.ready = true;
        B.made = true;
        B.uid = next_uid++;
    }
    void pick_lookup(PlIo& io, PW& w, const int32_t* chain0) {
        if (!c_.suffix || w.seq == 0) return;
        Lookup L;
        const int k = io.lookup_propose(2 * c_.S - 1, L);
        const int match = L.match;
        if (k <= 0 || L.tok[0] != chain0[0]) return;
        int t = 0;
        if (!io.lookup_choose(w.T, std::min(k, c_.S - 1), match, t)) return;
        w.T = t;
        for (int i = 1; i < w.T; ++i) {
            w.tok[i] = L.tok[i - 1];
            w.prob[i - 1] = 1.0f;
        }
        w.sfx = true;
        w.sfx_match = match;
        const int rest = k - (w.T - 1);
        if (&w == &A && rest >= 1 && c_.lookup_pon > 0.0f && !b_done) {
            B = PW{};
            B.seq = A.seq + 1;
            B.p = A.p + A.T;
            B.T = std::min(c_.S, rest);
            for (int i = 0; i < B.T; ++i) B.tok[i] = L.tok[A.T - 1 + i];
            for (int i = 1; i < B.T; ++i) B.prob[i - 1] = 1.0f;
            B.p_on = c_.lookup_pon;
            B.ready = true;
            B.made = true;
            B.uid = next_uid++;
            b_done = true;
            io.trace("CD", A.seq + 1, 3, (int) (1000.0f * B.p_on));
        }
    }
    bool vf0_in_flight(const PW& w) const {   // VF(0, w).in_flight(): a window of w's parity in flight on stage 0
        const int q = slot(w);
        auto on0 = [&](const PW& x) { return x.fl && x.fr == 0 && slot(x) == q; };
        return on0(A) || on0(B) || (doomed && on0(D));
    }
    static PlOp gap0() {   // a_gap(0): no window, no slot
        PlOp o;
        o.op = Op::Gap;
        o.st = 0;
        return o;
    }
    PlOp op(Op k, int st, const PW& w) const {
        PlOp o;
        o.op = k;
        o.st = st;
        o.slot = slot(w);
        o.seq = w.seq;
        o.uid = w.uid;
        return o;
    }

    Config c_;
    PW A, B, D, undo_w;
    bool doomed = false, ending = false, early_used = false, b_done = false;
    int chain_kind = 0, chain_n = 0, undo_keep = 0;
    int32_t undo_ple[2] = {-1, -1};
    int64_t pl_fm = 0;
    uint64_t next_uid = 1;
    int32_t outp[8] = {};
    double last_verdict = 0.0;
};

int OldLoop::step(PlIo& io) {
    if ((doomed && !pump0(io, D, true)) || !pump0(io, A, false) || !pump0(io, B, false)) return -1;
    if (A.s1 && !A.s1_done) {
        const int r = io.last_poll(slot(A), outp);
        if (r < 0) return -1;
        if (r == 1) {
            A.s1_done = true;
            io.trace("F1", A.seq, A.T, 0);
        }
    }
    if (doomed && !D.fl) {
        for (int st = 0; st < c_.n_front; ++st) {
            if ((undo_w.com & (1u << st)) == 0) continue;
            if (!ex(io, op(Op::Restore, st, undo_w))) return -1;
            if (st == 0) {
                PlOp q = op(Op::PleSet, 0, undo_w);
                q.ple[0] = undo_ple[0];
                q.ple[1] = undo_ple[1];
                if (!ex(io, q)) return -1;
            }
            PlOp m = op(Op::Commit, st, undo_w);
            m.keep = undo_keep;
            m.p = undo_w.p;
            if (!ex(io, m)) return -1;
        }
        doomed = false;
        io.trace("U", undo_w.seq, undo_keep, 0);
    }
    if (ending) return doomed ? 1 : 0;
    if (chain_kind != 0) {
        const int k_ready = io.chain_ready();
        if (k_ready < 0) return -1;
        if (chain_kind == 2 && !early_used) {
            const int k = k_ready;
            bool decided = k >= c_.S_mtp - 1;
            if (!decided && c_.spec_min_p > 0.0)
                for (int j = 0; j < k; ++j)
                    if (io.chain_prob()[j] < (float) c_.spec_min_p) { decided = true; break; }
            if (decided) {
                A.T = t_rule(io.chain_prob(), 0, std::min(k, c_.S_mtp - 1));
                for (int i = 1; i < A.T; ++i) {
                    A.tok[i] = io.chain_tok()[i - 1];
                    A.prob[i - 1] = io.chain_prob()[i - 1];
                }
                pick_lookup(io, A, io.chain_tok());
                A.ready = true;
                early_used = true;
                io.trace("CE", A.seq, A.T, 0);
            }
        }
        const int r = io.chain_poll();
        if (r < 0) return -1;
        const int k = r == 1 ? chain_n : k_ready;
        const int32_t* oc = io.chain_tok();
        const float* opr = io.chain_prob();
        if (r == 1 && chain_kind == 2 && !early_used) {
            A.T = t_rule(opr, 0, std::min(chain_n, c_.S_mtp - 1));
            for (int i = 1; i < A.T; ++i) {
                A.tok[i] = oc[i - 1];
                A.prob[i - 1] = opr[i - 1];
            }
            pick_lookup(io, A, oc);
            A.ready = true;
            early_used = true;
        }
        if (!b_done && (chain_kind == 1 || early_used)) {
            const int base = A.T - 1;
            bool decided = r == 1 || k >= base + c_.S_mtp;
            if (!decided && k > base + 1 && c_.spec_min_p > 0.0)
                for (int j = base + 1; j < k; ++j)
                    if (opr[j] < (float) c_.spec_min_p) { decided = true; break; }
            if (decided) {
                float pa = 1.0f;
                if (chain_kind == 1) {
                    bool agree = true;
                    for (int i = 0; i + 1 < A.T; ++i) {
                        if (oc[i] != A.tok[i + 1]) agree = false;
                        pa *= c_.agree ? opr[i] : A.prob[i];
                    }
                    if (c_.agree && !agree) pa = 0.0f;
                } else {
                    for (int i = 0; i + 1 < A.T; ++i) pa *= A.prob[i];
                }
                if (!A.sfx) make_b(oc, opr, base, k, pa);
                if (B.ready && B.p_on >= c_.theta && c_.prestage && !vf0_in_flight(B) && !doomed) {
                    int32_t prev[2];
                    io.ple_prev(prev);
                    for (int i = 0; i < A.T; ++i) {
                        prev[0] = prev[1];
                        prev[1] = A.tok[i];
                    }
                    PlOp o = op(Op::Prestage, 0, B);
                    o.T = B.T;
                    o.tok = B.tok;
                    o.p = B.p;
                    o.ple[0] = prev[0];
                    o.ple[1] = prev[1];
                    if (!ex(io, o)) return -1;
                }
                b_done = true;
                io.trace("CD", A.seq, chain_kind, (int) (1000.0f * B.p_on));
            }
        }
        if (r == 1) chain_kind = 0;
    }
    if (A.ready && !A.fl && A.fr < c_.n_front && !doomed) {
        const int st = A.fr;
        if (st == 0 && A.p + A.T > c_.max_context) {
            ending = true;
            return 1;
        }
        if (st == 0 && !ex(io, gap0())) return -1;
        if (!ex(io, op(Op::SnapTake, st, A))) return -1;
        PlOp l = op(Op::Launch, st, A);
        l.T = A.T;
        l.tok = A.tok;
        l.p = A.p;
        if (!ex(io, l)) return -1;
        A.fl = true;
        if (st == 0) A.launched = true;
        io.trace("L0", A.seq, A.T, st);
    }
    if (B.ready && !B.fl && B.fr < c_.n_front && B.fr < c_.spec_depth && A.fr > B.fr && !doomed) {
        const int st = B.fr;
        if (st > 0 || (B.p_on >= c_.theta && B.p + B.T <= c_.max_context)) {
            if (st == 0 && !ex(io, gap0())) return -1;
            if (!ex(io, op(Op::SnapWait, st, A))) return -1;
            if (st == 0) io.ple_prev(undo_ple);
            PlOp m = op(Op::Commit, st, A);
            m.keep = A.T;
            m.p = A.p;
            m.spec = true;
            if (!ex(io, m)) return -1;
            if (!ex(io, op(Op::SnapTake, st, B))) return -1;
            PlOp l = op(Op::Launch, st, B);
            l.T = B.T;
            l.tok = B.tok;
            l.p = B.p;
            if (!ex(io, l)) return -1;
            A.com |= 1u << st;
            B.fl = true;
            if (st == 0) {
                B.launched = true;
                B.spec = true;
            }
            io.trace("L0", B.seq, B.T, 10 + st);
        } else {
            B.ready = false;
        }
    }
    if (A.finished && !A.s1) {
        PlOp g;
        g.op = Op::Gap;
        g.st = c_.n_front;
        if (!ex(io, g)) return -1;
        PlOp l = op(Op::Launch, c_.n_front, A);
        l.T = A.T;
        l.tok = A.tok;
        l.p = A.p;
        if (!ex(io, l)) return -1;
        A.s1 = true;
        io.trace("L1", A.seq, A.T, 0);
    }
    if (!A.s1_done) return 1;
    if (chain_kind != 0 && io.chain_live()) {
        PlOp w;
        w.op = Op::ChainWaitIdle;
        if (!ex(io, w)) return -1;
    }
    if (chain_kind != 0 && !B.launched && !B.made) B = PW{};
    chain_kind = 0;
    int a = 0;
    while (a < A.T - 1 && A.tok[a + 1] == outp[a]) ++a;
    {
        PlOp m = op(Op::Commit, c_.n_front, A);
        m.keep = a + 1;
        m.p = A.p;
        if (!ex(io, m)) return -1;
    }
    bool last = false;
    if (!io.verdict(A, a, outp, last)) return -1;
    const bool on = B.launched && a == A.T - 1 && B.tok[0] == outp[A.T - 1];
    io.trace("V", A.seq, a, on ? 1 : (B.launched ? -1 : 0));
    {
        const double now = io.now_ms();
        if (A.seq > 0 && !last) io.observe(A, a, now - last_verdict);
        last_verdict = now;
    }
    if (!last) {
        PlOp o;
        o.op = Op::AdaptVerdict;
        if (!ex(io, o)) return -1;
    }
    if (on && !last) {
        chain_n = std::min(kMaxRows - 1, B.T + c_.S_mtp - 1);
        PlOp cl = op(Op::ChainLaunch, -1, A);
        cl.T = A.T;
        cl.outp = outp;
        cl.p = A.p;
        cl.a = a;
        cl.kind = 1;
        cl.n_force = B.T - 1;
        for (int i = 0; i < B.T - 1; ++i) cl.force[i] = B.tok[i + 1];
        cl.n_out = cl.n_early = chain_n;
        if (!ex(io, cl)) return -1;
        chain_kind = 1;
        io.trace("CL", A.seq, 1, 0);
        early_used = false;
        b_done = false;
        for (int st = 0; st < c_.n_front; ++st) {
            if ((A.com & (1u << st)) != 0) continue;
            PlOp m = op(Op::Commit, st, A);
            m.keep = A.T;
            m.p = A.p;
            if (!ex(io, m)) return -1;
        }
        A = B;
        B = PW{};
    } else {
        for (int st = 0; st < c_.n_front; ++st) {
            if ((A.com & (1u << st)) != 0) continue;
            PlOp m = op(Op::Commit, st, A);
            m.keep = a + 1;
            m.p = A.p;
            if (!ex(io, m)) return -1;
        }
        if (B.launched) {
            doomed = true;
            D = B;
            undo_w = A;
            undo_keep = a + 1;
        }
        B = PW{};
        if (!last) {
            chain_n = kMaxRows - 1;
            PlOp cl = op(Op::ChainLaunch, -1, A);
            cl.T = A.T;
            cl.outp = outp;
            cl.p = A.p;
            cl.a = a;
            cl.kind = 2;
            cl.n_out = cl.n_early = chain_n;
            if (!ex(io, cl)) return -1;
            chain_kind = 2;
            io.trace("CL", A.seq, 2, 0);
            early_used = false;
            b_done = false;
            PW nA;
            nA.seq = A.seq + 1;
            nA.p = A.p + a + 1;
            nA.tok[0] = outp[a];
            nA.uid = next_uid++;
            A = nA;
        }
    }
    if (!last) {
        PlOp o;
        o.op = Op::AdaptTick;
        if (!ex(io, o)) return -1;
    }
    if (last) ending = true;
    return 1;
}

// ---------------------------------------------------------------------------------------------------------------
// The recorded trace's verdicts: 'R p0 n', then n lines
// 'T a on bT'
// ---- trace data (generated; do not edit)
const char* const kTrace1017 =
    R"PLT(
R 58 10
1 0 0 0 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 4 0
4 3 1 2 0
2 0 0 0 0
1 0 -1 1 0
2 1 1 2 0
R 89 179
1 0 0 0 0
4 3 1 3 0
3 1 0 0 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 1 0
2 1 1 1 0
1 0 1 4 0
4 0 -1 1 0
4 3 -1 2 0
1 0 1 1 0
1 0 1 4 0
4 2 -1 3 0
1 0 1 2 0
2 1 -1 1 0
3 2 -1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 -1 1 0
1 0 -1 3 0
2 1 -1 1 0
1 0 1 4 0
4 3 1 4 0
4 1 -1 4 0
3 2 1 6 1
6 5 1 2 1
2 1 1 4 0
4 3 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 2 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 2 0
1 0 -1 3 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 2 0
2 0 -1 1 0
2 1 1 1 0
1 0 -1 3 0
1 0 -1 1 0
2 1 -1 1 0
2 1 -1 2 0
2 1 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
4 3 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 0 -1 1 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 1 0
3 2 -1 1 0
1 0 -1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 0 0 0
2 1 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
2 1 0 0 0
2 1 1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 0 -1 4 0
3 2 -1 1 0
1 0 1 1 0
1 0 -1 4 0
2 1 -1 2 0
2 1 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 1 4 0
4 3 1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 2 0
)PLT"
    R"PLT(
2 1 1 1 0
1 0 -1 2 0
2 1 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
4 3 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 1 0
2 1 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
R 86 305
1 0 0 0 0
4 3 1 3 0
3 2 0 0 0
4 3 1 4 0
4 2 0 0 0
3 2 1 6 1
6 5 0 0 1
6 5 1 6 1
6 5 1 2 1
2 1 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
3 2 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
3 2 1 1 0
1 0 0 0 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 4 0
2 1 -1 1 0
1 0 -1 3 0
4 3 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 -1 1 0
1 0 0 0 0
1 0 -1 3 0
1 0 -1 1 0
2 1 -1 1 0
3 2 -1 3 0
1 0 1 3 0
3 2 1 4 0
4 3 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 -1 3 0
2 1 1 3 0
3 2 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 2 0
2 1 1 2 0
2 1 0 0 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 -1 2 0
3 2 -1 1 0
4 3 1 4 0
4 3 1 1 0
1 0 -1 1 0
2 1 1 2 0
2 1 -1 2 0
1 0 -1 2 0
1 0 1 1 0
)PLT"
    R"PLT(
1 0 -1 1 0
3 2 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
3 2 1 2 0
2 1 -1 1 0
6 2 -1 6 1
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 4 0
4 3 1 2 0
2 0 0 0 0
4 3 1 2 0
2 1 -1 1 0
2 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
4 2 -1 2 0
1 0 1 1 0
1 0 0 0 0
2 1 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 0 0 0 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 -1 1 0
3 2 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 3 0
1 0 -1 2 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 3 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 3 0
3 2 -1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 2 0
6 0 -1 6 1
1 0 -1 1 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 2 0
1 0 -1 2 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
2 0 -1 3 0
2 1 1 2 0
2 1 -1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 3 0
3 2 1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 0 0 0
1 0 0 0 0
2 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
2 1 1 2 0
2 1 1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 1 2 0
2 1 -1 3 0
2 1 -1 1 0
2 0 -1 6 1
2 1 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 2 0
2 1 0 0 0
3 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
2 1 -1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 3 0
1 0 -1 2 0
2 1 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 0 0 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 2 0
2 1 -1 3 0
2 1 1 1 0
)PLT"
    R"PLT(
1 0 -1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 1 4 0
4 3 1 2 0
2 1 1 2 0
2 1 1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 2 0
2 1 1 4 0
4 3 1 4 0
4 2 0 0 0
1 0 1 1 0
1 0 -1 3 0
4 3 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 -1 2 0
4 3 1 2 0
2 1 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 2 0
2 1 -1 3 0
4 3 1 3 0
3 1 0 0 0
4 3 1 2 0
2 1 1 2 0
2 1 -1 2 0
4 3 1 4 0
4 3 1 2 0
2 1 1 1 0
1 0 1 2 0
2 1 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 4 0
4 3 1 1 0
1 0 -1 3 0
1 0 1 4 0
4 2 -1 3 0
R 83 302
1 0 0 0 0
1 0 -1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 -1 1 0
6 4 -1 6 1
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 2 0
2 1 0 0 0
1 0 1 1 0
1 0 -1 4 0
4 1 -1 3 0
2 1 -1 1 0
3 2 -1 1 0
3 1 -1 1 0
2 1 1 1 0
1 0 0 0 0
3 1 -1 1 0
1 0 1 1 0
1 0 -1 4 0
1 0 -1 2 0
2 1 -1 2 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 3 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 3 0
3 0 0 0 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
3 2 -1 3 0
4 3 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 3 0
4 1 -1 1 0
2 1 1 3 0
3 0 -1 1 0
1 0 1 2 0
2 1 -1 1 0
4 3 -1 1 0
2 1 1 1 0
1 0 -1 3 0
1 0 1 4 0
4 3 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 4 0
4 3 -1 4 0
2 1 1 4 0
4 3 1 4 0
4 0 -1 2 0
1 0 -1 1 0
4 3 1 3 0
3 2 -1 2 0
3 2 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
3 2 -1 3 0
1 0 -1 2 0
2 1 -1 2 0
2 1 -1 4 0
2 1 0 0 0
1 0 1 4 0
4 3 -1 3 0
1 0 1 2 0
2 1 -1 4 0
1 0 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 4 0
4 3 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
4 2 -1 1 0
1 0 1 2 0
2 1 -1 2 0
1 0 -1 1 0
3 2 -1 4 0
3 2 -1 4 0
3 2 1 1 0
1 0 -1 3 0
1 0 1 3 0
3 2 -1 1 0
1 0 1 1 0
1 0 -1 3 0
1 0 1 4 0
4 2 -1 1 0
4 3 1 2 0
)PLT"
    R"PLT(
2 1 1 2 0
2 1 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 -1 1 0
4 2 -1 1 0
2 1 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 0 0 0 0
1 0 -1 2 0
2 1 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 0 0 0
2 1 -1 2 0
2 1 1 3 0
3 2 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 4 0
2 1 -1 3 0
3 2 -1 1 0
1 0 1 1 0
1 0 -1 3 0
2 1 -1 4 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 4 0
1 0 1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
2 1 -1 1 0
2 1 -1 2 0
1 0 1 4 0
4 3 1 2 0
2 1 1 2 0
2 1 1 3 0
3 0 0 0 0
4 3 1 1 0
1 0 1 4 0
4 3 -1 1 0
4 3 1 4 0
4 1 0 0 0
2 1 1 3 0
3 2 1 1 0
1 0 1 2 0
2 1 1 4 0
4 3 -1 1 0
4 3 -1 2 0
1 0 -1 4 0
3 2 0 0 0
2 1 1 1 0
1 0 -1 1 0
2 1 -1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 4 0
4 3 -1 1 0
1 0 -1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
2 1 -1 1 0
4 3 1 2 0
2 1 1 4 0
4 3 -1 2 0
2 1 -1 4 0
4 3 1 2 0
2 1 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 0 -1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 0 0 0
2 1 -1 2 0
1 0 -1 1 0
4 3 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
)PLT"
    R"PLT(
1 0 1 2 0
2 1 1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 1 2 0
2 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 3 0
3 2 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 3 0
3 0 -1 1 0
1 0 -1 1 0
1 0 -1 3 0
3 0 -1 1 0
1 0 0 0 0
1 0 1 3 0
3 0 0 0 0
2 1 1 2 0
2 0 -1 1 0
1 0 1 4 0
4 0 -1 1 0
2 1 -1 4 0
3 0 -1 6 1
4 3 1 1 0
1 0 1 1 0
1 0 -1 1 0
4 3 1 4 0
4 3 1 3 0
3 2 1 3 0
3 2 0 0 0
2 1 1 1 0
1 0 -1 2 0
1 0 -1 4 0
1 0 1 3 0
3 2 0 0 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 1 0
3 2 1 1 0
1 0 0 0 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 2 0
2 1 -1 4 0
R 26368 253
1 0 0 0 0
4 2 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 1 2 0
2 1 -1 3 0
1 0 1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 -1 4 0
2 1 -1 6 1
1 0 -1 2 0
2 1 1 1 0
1 0 -1 1 0
2 0 -1 3 0
2 1 -1 1 0
1 0 -1 1 0
1 0 1 2 0
2 0 0 0 0
1 0 -1 3 0
2 0 -1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
3 1 -1 2 0
1 0 -1 1 0
2 1 -1 1 0
1 0 0 0 0
1 0 -1 1 0
3 2 -1 1 0
1 0 0 0 0
1 0 -1 2 0
1 0 -1 1 0
3 2 -1 3 0
1 0 -1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 -1 1 0
3 2 1 2 0
2 0 -1 4 0
4 2 -1 2 0
1 0 -1 1 0
2 1 -1 2 0
1 0 -1 3 0
2 1 -1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
1 0 0 0 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
4 3 1 1 0
1 0 1 4 0
4 0 0 0 0
4 2 -1 3 0
3 2 -1 3 0
2 1 1 1 0
1 0 -1 2 0
2 0 -1 1 0
3 2 1 1 0
1 0 -1 2 0
2 1 1 2 0
2 1 0 0 0
3 2 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
3 1 -1 2 0
3 0 -1 2 0
4 3 -1 4 0
4 1 -1 1 0
1 0 -1 4 0
)PLT"
    R"PLT(
2 1 -1 3 0
3 2 -1 1 0
3 2 1 1 0
1 0 -1 2 0
1 0 -1 4 0
4 0 -1 1 0
1 0 -1 1 0
3 2 -1 1 0
3 2 -1 1 0
4 3 -1 3 0
3 2 -1 3 0
1 0 -1 1 0
1 0 -1 2 0
1 0 1 1 0
1 0 -1 3 0
3 1 -1 1 0
4 3 -1 3 0
3 2 -1 1 0
1 0 -1 2 0
3 2 1 1 0
1 0 -1 1 0
4 3 -1 2 0
1 0 1 2 0
2 1 1 4 0
4 0 -1 4 0
1 0 -1 2 0
2 1 1 2 0
2 0 -1 2 0
2 1 1 2 0
2 1 1 2 0
2 1 0 0 0
1 0 -1 1 0
2 0 -1 2 0
2 1 1 1 0
1 0 -1 3 0
2 0 -1 3 0
2 1 -1 3 0
2 0 0 0 0
2 1 0 0 0
4 1 -1 1 0
2 1 0 0 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 2 0
2 1 -1 2 0
3 2 -1 2 0
1 0 1 4 0
4 1 -1 1 0
2 0 -1 2 0
3 0 -1 2 0
2 1 1 3 0
3 2 -1 1 0
2 0 -1 2 0
4 3 -1 1 0
2 1 -1 3 0
4 0 -1 2 0
4 2 -1 2 0
1 0 -1 1 0
2 0 -1 2 0
1 0 -1 1 0
1 0 -1 2 0
2 0 -1 4 0
2 1 -1 4 0
1 0 1 1 0
1 0 -1 2 0
1 0 -1 1 0
2 1 -1 2 0
2 1 -1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 2 0
1 0 -1 2 0
1 0 1 4 0
4 2 0 0 0
1 0 1 1 0
1 0 1 2 0
2 1 -1 1 0
1 0 -1 1 0
1 0 -1 4 0
4 3 1 1 0
1 0 1 3 0
3 1 0 0 0
1 0 1 3 0
3 2 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 4 0
4 3 1 3 0
3 2 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 1 4 0
4 3 1 3 0
3 2 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 2 0
3 2 0 0 0
1 0 -1 1 0
1 0 1 2 0
2 1 -1 2 0
1 0 -1 1 0
1 0 -1 1 0
3 2 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 -1 1 0
2 1 -1 1 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 2 0
2 1 -1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 -1 1 0
3 2 -1 2 0
2 1 -1 1 0
3 2 0 0 0
)PLT"
    R"PLT(
1 0 1 1 0
1 0 -1 3 0
1 0 1 1 0
1 0 1 3 0
R 114702 231
1 0 0 0 0
2 1 -1 1 0
2 1 1 2 0
2 1 1 4 0
4 2 0 0 0
6 5 1 6 1
6 4 0 0 1
1 0 1 1 0
1 0 -1 1 0
1 0 0 0 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 -1 4 0
4 3 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 2 0
2 1 -1 1 0
2 1 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
1 0 1 2 0
2 0 0 0 0
2 1 -1 1 0
2 1 1 1 0
1 0 1 2 0
2 1 1 4 0
4 3 -1 4 0
1 0 -1 1 0
2 0 -1 1 0
2 1 1 2 0
2 1 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 1 2 0
2 1 1 2 0
2 1 1 2 0
2 0 -1 2 0
4 1 -1 1 0
2 1 1 2 0
2 1 -1 1 0
3 2 -1 1 0
2 1 0 0 0
1 0 1 3 0
3 2 1 4 0
4 3 1 3 0
3 2 -1 3 0
2 1 -1 2 0
4 3 0 0 0
2 1 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 -1 1 0
2 1 -1 1 0
2 1 -1 1 0
2 0 -1 4 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 3 0
1 0 1 1 0
1 0 1 2 0
2 1 1 2 0
2 1 1 1 0
1 0 1 4 0
4 1 -1 1 0
4 3 1 4 0
4 3 1 1 0
1 0 0 0 0
1 0 -1 2 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
2 1 -1 2 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 4 0
4 1 -1 4 0
1 0 -1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 1 0
1 0 1 1 0
1 0 0 0 0
1 0 -1 2 0
3 2 -1 1 0
1 0 -1 1 0
2 1 1 1 0
1 0 1 4 0
4 0 -1 4 0
4 3 -1 2 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 2 0
1 0 1 1 0
1 0 0 0 0
1 0 1 4 0
4 0 -1 4 0
4 3 1 3 0
3 2 1 1 0
1 0 -1 1 0
4 3 -1 1 0
2 1 1 3 0
3 2 -1 3 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 3 0
1 0 1 1 0
1 0 1 4 0
4 0 -1 1 0
1 0 -1 2 0
4 3 1 4 0
4 3 1 2 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
1 0 1 1 0
)PLT"
    R"PLT(
1 0 1 4 0
4 0 -1 2 0
2 1 -1 3 0
1 0 0 0 0
3 2 1 1 0
1 0 1 1 0
1 0 1 3 0
3 2 -1 2 0
3 2 -1 6 1
1 0 1 1 0
1 0 -1 1 0
1 0 -1 1 0
3 1 -1 1 0
4 3 1 3 0
3 1 -1 2 0
4 3 -1 1 0
2 1 1 1 0
1 0 1 1 0
1 0 -1 1 0
2 1 1 2 0
2 1 -1 1 0
2 0 -1 1 0
1 0 -1 4 0
4 0 -1 4 0
1 0 -1 4 0
1 0 -1 2 0
3 2 -1 4 0
4 2 -1 2 0
2 1 -1 1 0
2 1 -1 1 0
1 0 -1 1 0
2 1 -1 4 0
4 1 -1 4 0
1 0 -1 3 0
4 3 1 1 0
1 0 -1 3 0
4 0 -1 2 0
4 3 1 1 0
1 0 1 3 0
3 2 -1 2 0
2 1 1 1 0
1 0 1 2 0
2 0 0 0 0
4 1 0 0 0
1 0 -1 1 0
2 1 -1 2 0
6 5 -1 6 1
2 1 1 1 0
1 0 1 1 0
1 0 1 1 0
1 0 -1 4 0
4 1 -1 4 0
1 0 1 3 0
3 0 0 0 0
3 2 1 6 1
6 5 0 0 1
1 0 -1 1 0
6 1 -1 6 1
4 3 -1 1 0
1 0 -1 4 0
1 0 1 4 0
4 3 1 1 0
1 0 1 2 0
2 1 1 2 0
2 1 1 1 0
1 0 1 3 0
3 1 0 0 0
3 2 1 2 0
2 1 1 2 0
2 0 -1 1 0
4 3 -1 4 0
1 0 -1 3 0
2 1 -1 3 0
4 1 -1 4 0
2 1 -1 4 0
1 0 -1 4 0
4 1 -1 2 0
1 0 -1 4 0
4 3 -1 2 0
2 0 -1 3 0
4 1 -1 4 0
1 0 -1 1 0
2 1 -1 2 0
1 0 -1 4 0
2 1 -1 4 0
4 3 1 3 0
)PLT";
// ---- end trace data

struct TraceReq {
    int64_t p0 = 0;
    struct Rec { int T, a, on, bT, sfx; };
    std::vector<Rec> v;
};

std::vector<TraceReq> parse_trace(const char* s) {
    std::vector<TraceReq> out;
    while (*s) {
        while (*s == '\n' || *s == '\r' || *s == ' ') ++s;
        if (!*s) break;
        if (*s == 'R') {
            char* e = nullptr;
            TraceReq r;
            r.p0 = std::strtoll(s + 1, &e, 10);
            (void) std::strtol(e, &e, 10);
            out.push_back(r);
            s = e;
        } else {
            int f[5] = {};
            for (int k = 0; k < 5; ++k) {
                while (*s == ' ') ++s;
                char* e = nullptr;
                f[k] = (int) std::strtol(s, &e, 10);
                s = e;
            }
            out.back().v.push_back({f[0], f[1], f[2], f[3], f[4]});
        }
    }
    return out;
}

// default guesses everywhere (a hash): right 60%, probabilities high/mid/low 70/20/10 (right) or 50/20/30 (wrong)
void fill_default(World& w, uint64_t salt) {
    for (size_t i = 0; i < w.ok.size(); ++i) {
        const uint64_t h = mix64(salt * 1000003u + i);
        w.ok[i] = (h % 100) < 60;
        const uint64_t g = (h >> 20) % 10;
        w.pr[i] = w.ok[i] ? (g < 7 ? kHigh : g < 9 ? kMid : kLow) : (g < 5 ? kHigh : g < 7 ? kMid : kLow);
        w.okw[i] = 0;
        const uint64_t f = (h >> 40) % 10;
        w.prw[i] = f < 5 ? kHigh : f < 7 ? kMid : kLow;
    }
}

// the world that replays a trace request: every verified window's drafts right up to its a, the next one wrong, its
// size from the probabilities (high inside, mid/low where it ends: mid where a window was launched behind it)
World world_from_trace(const TraceReq& r, uint64_t salt) {
    World w;
    w.p0 = r.p0;
    w.salt = salt;
    int64_t n = 64, emitted = 0;
    for (const auto& x : r.v) {
        n += x.a + 1 + x.T + x.bT + 2;
        emitted += x.a + 1;
    }
    w.size(n);
    fill_default(w, salt);
    auto setc = [&](int64_t q, bool ok, float pr) {
        if (!w.in(q)) return;
        w.ok[w.ix(q)] = ok;
        w.pr[w.ix(q)] = pr;
        w.set_c[w.ix(q)] = 1;
    };
    auto setw = [&](int64_t q, bool ok, float pr) {
        if (!w.in(q) || w.set_w[w.ix(q)]) return;
        w.okw[w.ix(q)] = ok;
        w.prw[w.ix(q)] = pr;
        w.set_w[w.ix(q)] = 1;
    };
    int64_t p = r.p0;
    for (const auto& x : r.v) {
        for (int j = 1; j <= x.a; ++j) setc(p + j, true, kHigh);
        const float end_pr = x.on != 0 ? kMid : kLow;
        if (x.a < x.T - 1) {
            setc(p + x.a + 1, false, kHigh);
            for (int j = x.a + 2; j < x.T; ++j) setw(p + j, false, kHigh);
            setw(p + x.T, false, end_pr);
        } else {
            setc(p + x.T, x.on == 1, end_pr);
        }
        if (x.sfx && w.in(p)) {   // a lookup window: the lookup offered here, taken at its size; its continuation
            w.lk[w.ix(p)] = 1;     // past the window right as far as the trace's verdict says
            w.lk_t[w.ix(p)] = (uint8_t) x.T;
            for (int j = 1; j <= x.T + 8; ++j)
                if (w.in(p + j)) w.lkok[w.ix(p + j)] = j <= x.a || (j == x.T && x.on == 1) || j > x.T;
            w.lookup_t = 4;
        }
        if (x.on == -1 && x.bT > 0) {
            const int64_t bp = p + x.T;
            for (int j = 1; j < x.bT; ++j) setw(bp + j, false, kHigh);
            setw(bp + x.bT, false, kLow);
        }
        p += x.a + 1;
    }
    w.max_new = emitted;
    return w;
}

// ---------------------------------------------------------------------------------------------------------------
struct RunResult {
    bool ok = true;
    std::string err;
    int64_t verdicts = 0, produced = 0, iters = 0;
    double t_end = 0.0;
    Stats st;
    std::vector<std::string> log;
    std::vector<Sim::Ev> events;
    int64_t clamps = 0, restage_launch = 0, hazards_doomed = 0;
    int64_t end = 0;
};

RunResult run_model(const World& w, const Config& c, const Timing& tm, bool keep_log, bool old) {
    RunResult R;
    Sim sim(w, c, tm);
    sim.keep_log = keep_log;
    std::unordered_set<uint64_t> doomed_ever;
    sim.ever_doomed = [&](uint64_t u) { return doomed_ever.count(u) > 0; };
    Sched s;
    OldLoop o;
    int32_t ple[2];
    sim.ple_prev(ple);
    const int32_t x = w.truth(w.p0);
    std::string e;
    if (old) o.begin(c, w.p0, x, 0.0);
    else if (!s.begin(c, w.p0, x, ple, 0.0, e)) {
        R.ok = false;
        R.err = e;
        return R;
    }
    auto bad = [&](const std::string& m) {
        R.ok = false;
        R.err = m;
        R.log = std::move(sim.log);
        R.events = std::move(sim.events);
        return R;
    };
    const int64_t limit = 400000 + 4000 * (w.max_new + 10);
    for (int64_t it = 0;; ++it) {
        sim.progress = 0;
        const int r = old ? o.step(sim) : s.step(sim);
        if (old) {
            if (o.doomed_uid() != 0) doomed_ever.insert(o.doomed_uid());
        } else {
            for (int d = 0; d < s.ndoom(); ++d) doomed_ever.insert(s.doom(d).uid);
        }
        if (!sim.err.empty()) return bad("sim: " + sim.err + (old || s.err().empty() ? "" : " / " + s.err()));
        if (r < 0) return bad(old ? o.err : s.err());
        if (r == 0) {
            R.iters = it;
            break;
        }
        if (sim.progress == 0) {
            const double nx = sim.next_gpu_time();
            if (nx == kInf) return bad("deadlock: the loop waits and nothing can happen on the GPU");
            sim.t = std::max(sim.t + tm.host_iter, nx);
        } else {
            sim.t += tm.host_iter;
        }
        if (it > limit) return bad("the loop never ended");
    }
    std::string why;
    if (!sim.drain(why)) return bad(why);
    if (!sim.err.empty()) return bad("sim (drain): " + sim.err);
    // every reply token was the serial loop's (checked as emitted); the request ended where it should
    const bool ended = sim.eos || sim.produced >= w.max_new || (w.cancel_at >= 0 && sim.verdicts > w.cancel_at);
    if (!ended && sim.emit_end + c.S_mtp + 1 <= w.max_context)
        return bad("the loop ended early (" + std::to_string(sim.produced) + " of " + std::to_string(w.max_new) + ")");
    if (sim.verdicts > 0 && !sim.check_state(sim.last_end, why))
        return bad("CHECK_STATE: " + why + " differs from the serial loop's");
    R.verdicts = sim.verdicts;
    R.produced = sim.produced;
    R.t_end = sim.t;
    if (!old) R.st = s.stats();
    R.log = std::move(sim.log);
    R.events = std::move(sim.events);
    R.clamps = sim.clamps;
    R.restage_launch = sim.restage_launch;
    R.hazards_doomed = sim.hazards_doomed;
    R.end = sim.last_end;
    return R;
}

// ---------------------------------------------------------------------------------------------------------------
struct Tally {
    int64_t runs = 0, fails = 0, verdicts = 0, tokens = 0, clamps = 0, hazards_doomed = 0, restage_launch = 0;
    double ms = 0.0;
    Stats st;
    std::string first_err;
    void add(const RunResult& r) {
        ++runs;
        if (!r.ok) {
            ++fails;
            if (first_err.empty()) first_err = r.err;
            return;
        }
        verdicts += r.verdicts;
        tokens += r.produced;
        ms += r.t_end;
        clamps += r.clamps;
        hazards_doomed += r.hazards_doomed;
        restage_launch += r.restage_launch;
        const Stats& s = r.st;
        st.windows += s.windows;
        st.spec += s.spec;
        st.on += s.on;
        st.undo += s.undo;
        st.gate += s.gate;
        st.disagree += s.disagree;
        st.late += s.late;
        for (int i = 0; i < 2; ++i) {
            st.held_n[i] += s.held_n[i];
            st.held_ms[i] += s.held_ms[i];
            st.cls_n[i] += s.cls_n[i];
            st.cls_ms[i] += s.cls_ms[i];
        }
        st.c_made += s.c_made;
        st.c_launched += s.c_launched;
        st.c_on += s.c_on;
        st.c_doomed += s.c_doomed;
        st.undo_jobs += s.undo_jobs;
        st.stage_waits += s.stage_waits;
        st.stage_wait_ms += s.stage_wait_ms;
        st.slot_waits += s.slot_waits;
        st.slot_wait_ms += s.slot_wait_ms;
        st.restage += s.restage;
        st.restage_spec += s.restage_spec;
        st.chain_waits += s.chain_waits;
        st.chain_wait_ms += s.chain_wait_ms;
        st.chain_late += s.chain_late;
        st.chain_steps += s.chain_steps;
        st.chain_launches += s.chain_launches;
        st.chain_cancel += s.chain_cancel;
        for (int i = 0; i < kCaseN; ++i) st.case_hits[i] += s.case_hits[i];
    }
};

void print_tally(const char* name, Mode m, const Tally& t) {
    const Stats& s = t.st;
    std::printf("  %-3s %-11s runs %7lld fails %lld | verdicts %9lld tokens %9lld | sim %.1f tok/s | held %lld+%lld "
                "(H1 %.2f H2 %.2f ms) fresh %.2f ms\n",
                mode_name(m), name, (long long) t.runs, (long long) t.fails, (long long) t.verdicts,
                (long long) t.tokens, t.ms > 0 ? 1000.0 * (double) t.tokens / t.ms : 0.0, (long long) s.held_n[0],
                (long long) s.held_n[1], s.held_n[0] ? s.held_ms[0] / (double) s.held_n[0] : 0.0,
                s.held_n[1] ? s.held_ms[1] / (double) s.held_n[1] : 0.0,
                s.cls_n[0] > 0 ? s.cls_ms[0] / s.cls_n[0] : 0.0);
    std::printf("      C made %lld launched %lld on %lld doomed %lld | undo jobs %lld stage-waits %lld (%.1f ms) | slot "
                "waits %lld (%.1f ms) | restage %lld (spec %lld; sim %lld) | chain waits %lld (%.1f ms) late %lld "
                "cancelled %lld | chain steps %.2f/verdict | doomed hand-offs overwritten %lld | chain clamps %lld\n",
                (long long) s.c_made, (long long) s.c_launched, (long long) s.c_on, (long long) s.c_doomed,
                (long long) s.undo_jobs, (long long) s.stage_waits, s.stage_wait_ms, (long long) s.slot_waits,
                s.slot_wait_ms, (long long) s.restage, (long long) s.restage_spec, (long long) t.restage_launch,
                (long long) s.chain_waits, s.chain_wait_ms, (long long) s.chain_late, (long long) s.chain_cancel,
                t.verdicts ? (double) s.chain_steps / (double) t.verdicts : 0.0, (long long) t.hazards_doomed,
                (long long) t.clamps);
    std::printf("      cases:");
    for (int i = 0; i < kCaseN; ++i)
        if (s.case_hits[i] > 0) std::printf(" [%s] %lld", case_name(i), (long long) s.case_hits[i]);
    std::printf("\n");
    if (t.fails > 0) std::printf("      FIRST FAILURE: %s\n", t.first_err.c_str());
}

Config base_config(Mode m) {
    Config c;
    c.mode = m;
    c.n_front = 3;
    c.S = 6;
    c.S_mtp = 4;
    c.spec_min_p = 0.85;
    c.theta = 0.2f;
    c.theta3 = -1.0f;
    c.chain_out2 = 7;
    c.chain_max = 11;
    c.chain_stop = true;
    c.max_context = (int64_t) 1 << 40;
    return c;
}

const Mode kModes[3] = {Mode::N2, Mode::B0, Mode::B};

// ---- (d) N2 against the old loop on one world: the same operations, events and host times, line for line
struct Compare {
    int64_t runs = 0, same = 0, diff = 0;
    std::string first;
};
void compare_n2_old(const World& w, const Config& c0, const Timing& tm0, Compare& cmp, const char* what) {
    Config c = c0;
    c.mode = Mode::N2;
    Timing tm = tm0;
    tm.clamp = false;   // the old loop's drafter: no clamp at the context's end
    const RunResult a = run_model(w, c, tm, true, true);
    const RunResult b = run_model(w, c, tm, true, false);
    ++cmp.runs;
    bool same = a.ok == b.ok && a.log.size() == b.log.size();
    size_t i = 0;
    for (; i < std::min(a.log.size(), b.log.size()); ++i)
        if (a.log[i] != b.log[i]) {
            same = false;
            break;
        }
    if (same && !a.ok && a.err != b.err) same = false;
    if (same) {
        ++cmp.same;
        return;
    }
    ++cmp.diff;
    if (cmp.first.empty()) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "%s: old %s / N2 %s, logs %zu vs %zu lines, first difference at line %zu\n", what,
                      a.ok ? "ok" : a.err.c_str(), b.ok ? "ok" : b.err.c_str(), a.log.size(), b.log.size(), i);
        cmp.first = buf;
        const size_t from = i > 6 ? i - 6 : 0;
        for (size_t j = from; j < std::min(std::max(a.log.size(), b.log.size()), i + 4); ++j) {
            cmp.first += "    old: " + (j < a.log.size() ? a.log[j] : std::string("-")) + "\n";
            cmp.first += "    N2 : " + (j < b.log.size() ? b.log[j] : std::string("-")) + "\n";
        }
    }
}

// ---- (b) random worlds
struct RandomCase {
    World w;
    Config c;
    Timing tm;
};
bool g_stress = false;   // --stress: every random world with crawling snapshot and source copies
RandomCase make_random(uint64_t seed) {
    static const int kFm[3] = {2, 3, 7}, kFmc[2] = {2, 3};
    static const double kPc[7] = {0.0, 0.3, 0.5, 0.8, 1.0, 0.6, 0.7}, kJit[3] = {0.0, 0.05, 0.3},
                        kCs[4] = {0.3, 1.0, 1.0, 2.5};
    Rng r(seed);
    RandomCase rc;
    Config& c = rc.c;
    c = base_config(Mode::N2);
    c.n_front = r.p(0.75) ? 3 : r.p(0.5) ? 2 : 1;
    c.S_mtp = r.p(0.6) ? 4 : 1 + r.i(4);
    c.S = std::min(8, c.S_mtp + 2);
    c.spec_min_p = r.p(0.8) ? 0.85 : 0.0;
    c.theta = r.p(0.7) ? 0.2f : r.p(0.5) ? 0.0f : 0.5f;
    c.theta3 = r.p(0.7) ? -1.0f : 0.35f;
    c.force_miss = r.p(0.7) ? 0 : kFm[r.i(3)];
    c.force_miss_c = r.p(0.7) ? 0 : kFmc[r.i(2)];
    c.agree = r.p(0.9);
    c.prestage = r.p(0.92);
    c.suffix = r.p(0.3);
    c.chain_stop = r.p(0.7);
    c.slots = r.p(0.85) ? 0 : 5;
    World& w = rc.w;
    w.p0 = 16 + r.i(4000);
    w.salt = mix64(seed);
    const double pc = kPc[r.i(7)];
    const int64_t n = 64 + 3 * 900;
    w.size(n);
    for (int64_t i = 0; i < n; ++i) {
        const bool ok = r.p(pc);
        w.ok[(size_t) i] = ok;
        const double u = r.u();
        w.pr[(size_t) i] = ok ? (u < 0.75 ? kHigh : u < 0.95 ? kMid : kLow) : (u < 0.35 ? kHigh : u < 0.7 ? kMid : kLow);
        w.okw[(size_t) i] = r.p(0.1);
        const double v = r.u();
        w.prw[(size_t) i] = v < 0.5 ? kHigh : v < 0.7 ? kMid : kLow;
        w.lk[(size_t) i] = c.suffix && r.p(0.06);
        w.lkok[(size_t) i] = r.p(0.85);
    }
    if (c.suffix) w.lookup_t = r.p(0.5) ? c.S : 4;
    w.max_new = r.p(0.2) ? 1 + r.i(16) : 16 + r.i(800);
    if (r.p(0.15)) w.eos_pos = w.p0 + 1 + r.i((int) std::min<int64_t>(n - 64, w.max_new + 5));
    if (r.p(0.12)) w.cancel_at = r.i(40);
    if (r.p(0.15)) w.max_context = w.p0 + 1 + r.i((int) std::min<int64_t>(n - 64, w.max_new + 20));
    c.max_context = w.max_context;
    Timing& tm = rc.tm;
    tm.seed = mix64(seed + 5);
    tm.jitter = kJit[r.i(3)];
    const double cs = kCs[r.i(4)];
    tm.chain_step *= cs;
    tm.chain_round *= cs;
    if (r.p(0.15) || g_stress) tm.snap_slow = 0.3;
    if (r.p(0.15) || g_stress) tm.src_slow = g_stress ? 0.6 : 0.3;
    tm.chain_n_engine = (mix64(seed + 11) & 3) != 0;   // a quarter with the old reading (not from r: the worlds stay as they were)
    if (g_stress) c.slots = 0;   // the fewest slots: windows wait for one, and take a slot the moment it is free
    return rc;
}

// ---- (c) exhaustive small worlds: every combination of 5 positions' guess (right/wrong) x probability
// (high/mid/low), repeated along the text
void make_small(World& w, int pattern, uint64_t salt) {
    w = World{};
    w.p0 = 40;
    w.salt = salt;
    w.size(120);
    for (int64_t i = 0; i < 120; ++i) {
        int code = pattern;
        for (int k = 0; k < (int) (i % 5); ++k) code /= 6;
        code %= 6;
        w.ok[(size_t) i] = code < 3;
        w.pr[(size_t) i] = code % 3 == 0 ? kHigh : code % 3 == 1 ? kMid : kLow;
        w.okw[(size_t) i] = 0;
        w.prw[(size_t) i] = (i % 3) == 0 ? kLow : kHigh;
    }
}

int64_t g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    int64_t random_target = 1000000;
    std::string only;
    uint64_t seed = 1;
    int patterns = 7776;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--random" && i + 1 < argc) random_target = std::atoll(argv[++i]);
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--patterns" && i + 1 < argc) patterns = std::atoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--stress") g_stress = true;
    }
    if (quick) {
        random_target = std::min<int64_t>(random_target, 60000);
        patterns = std::min(patterns, 400);
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Tally all[3];
    Compare cmp;

    // ---- (a) the recorded trace
    if (only.empty() || only == "a") {
        const std::vector<TraceReq> tr = parse_trace(kTrace1017);
        int64_t nv = 0;
        for (const auto& r : tr) nv += (int64_t) r.v.size();
        std::printf("(a) the recorded trace: %zu requests, %lld verdicts\n", tr.size(), (long long) nv);
        Tally ta[3];
        int64_t v_same = 0, v_tot = 0, req_same = 0;
        for (size_t i = 0; i < tr.size(); ++i) {
            const World w = world_from_trace(tr[i], 1000 + i);
            Timing tm;
            tm.seed = 77 + i;
            tm.chain_step = 1.45;
            for (int m = 0; m < 3; ++m) {
                Config c = base_config(kModes[m]);
                c.suffix = true;   // the engine's default suffix_draft 3: the trace has lookup windows
                const RunResult r = run_model(w, c, tm, false, false);
                ta[m].add(r);
                all[m].add(r);
                if (!r.ok) std::printf("    trace #%zu %s: %s\n", i, mode_name(kModes[m]), r.err.c_str());
                if (m == 0 && r.ok) {   // N2's verdicts against the trace's (how close the replay is)
                    size_t k = 0;
                    bool same_req = true;
                    for (const auto& e : r.events) {
                        if (e.ev != "V") continue;
                        if (k < tr[i].v.size()) {
                            const auto& x = tr[i].v[k];
                            const bool s = x.a == e.x0 && x.on == e.x1;
                            v_same += s ? 1 : 0;
                            same_req = same_req && s;
                        }
                        ++k;
                    }
                    if (verbose && !same_req) {   // the first verdict the replay does not reproduce
                        size_t kk = 0, at = 0;
                        std::vector<const Sim::Ev*> vs;
                        for (const auto& e : r.events)
                            if (e.ev == "V") vs.push_back(&e);
                        for (; kk < vs.size() && kk < tr[i].v.size(); ++kk)
                            if (tr[i].v[kk].a != vs[kk]->x0 || tr[i].v[kk].on != vs[kk]->x1) break;
                        at = kk;
                        std::printf("    request %zu: first different verdict #%zu\n", i, at);
                        for (size_t j = at > 3 ? at - 3 : 0; j < std::min(vs.size(), at + 3); ++j)
                            std::printf("      #%zu trace T %d a %d on %d bT %d | replay seq %d a %d on %d\n", j,
                                        j < tr[i].v.size() ? tr[i].v[j].T : -1, j < tr[i].v.size() ? tr[i].v[j].a : -1,
                                        j < tr[i].v.size() ? tr[i].v[j].on : -9, j < tr[i].v.size() ? tr[i].v[j].bT : -1,
                                        vs[j]->seq, vs[j]->x0, vs[j]->x1);
                    }
                    v_tot += (int64_t) tr[i].v.size();
                    same_req = same_req && k == tr[i].v.size();
                    req_same += same_req ? 1 : 0;
                }
            }
            Config cn = base_config(Mode::N2);
            cn.suffix = true;
            compare_n2_old(w, cn, tm, cmp, "trace");
        }
        for (int m = 0; m < 3; ++m) {
            print_tally("trace", kModes[m], ta[m]);
            check(ta[m].fails == 0, std::string("(a) trace, mode ") + mode_name(kModes[m]));
        }
        // the symbolic GPU's timing (the recorded trace's stage times, no contention): depth 3 must actually overlap the guesses
        auto tps = [](const Tally& t) { return t.ms > 0.0 ? 1000.0 * (double) t.tokens / t.ms : 0.0; };
        std::printf("  simulated tok/s: N2 %.2f, B0 %+.1f%%, B %+.1f%%\n", tps(ta[0]), 100.0 * (tps(ta[1]) / tps(ta[0]) - 1.0),
                    100.0 * (tps(ta[2]) / tps(ta[0]) - 1.0));
        check(tps(ta[1]) >= 0.995 * tps(ta[0]), "(a) trace: B0 slower than N2 in the simulation");
        check(tps(ta[2]) >= 1.04 * tps(ta[0]), "(a) trace: B less than 4% faster than N2 in the simulation");
        std::printf("  N2 replay against the trace: %lld of %lld verdicts the same (a, on), %lld of %zu requests whole\n",
                    (long long) v_same, (long long) v_tot, (long long) req_same, tr.size());
        std::printf("  [%.1f s]\n", secs());
    }

    // ---- (b) random worlds
    if (only.empty() || only == "b") {
        std::printf("(b) random worlds: until %lld verdicts per mode\n", (long long) random_target);
        Tally tb[3];
        int64_t req = 0;
        for (uint64_t k = 0; tb[2].verdicts < random_target && k < (uint64_t) random_target; ++k, ++req) {
            const RandomCase rc = make_random(seed * 1000003u + k);
            for (int m = 0; m < 3; ++m) {
                Config c = rc.c;
                c.mode = kModes[m];
                if (kModes[m] == Mode::B && c.n_front < 2) c.n_front = 2;   // depth 3 wants three stages or more
                Timing tm = rc.tm;
                if (kModes[m] == Mode::N2) tm.src_slow = 0.0;
                const RunResult r = run_model(rc.w, c, tm, false, false);
                tb[m].add(r);
                all[m].add(r);
                if (!r.ok && tb[m].fails <= 3)
                    std::printf("    random #%llu %s: %s\n", (unsigned long long) k, mode_name(kModes[m]), r.err.c_str());
            }
            if (k < (quick ? 300u : 3000u)) {
                Timing tm = rc.tm;
                tm.src_slow = 0.0;
                compare_n2_old(rc.w, rc.c, tm, cmp, "random");
            }
        }
        for (int m = 0; m < 3; ++m) {
            print_tally("random", kModes[m], tb[m]);
            check(tb[m].fails == 0, std::string("(b) random, mode ") + mode_name(kModes[m]));
        }
        std::printf("  %lld requests [%.1f s]\n", (long long) req, secs());
    }

    // ---- (c) exhaustive small worlds
    if (only.empty() || only == "c") {
        std::printf("(c) exhaustive: %d worlds x endings (max_new 1..12, cancel after each of verdicts 0..5, EOS at "
                    "each of 8 positions, the context edge at 10 places) x 2 timings x 2 configs\n", patterns);
        Tally tc[3];
        for (int pat = 0; pat < patterns; ++pat) {
            World w0;
            make_small(w0, pat, 9000 + (uint64_t) pat);
            std::vector<World> ws;
            for (int mn = 1; mn <= 12; ++mn) {
                World w = w0;
                w.max_new = mn;
                ws.push_back(w);
            }
            for (int ca = 0; ca <= 5; ++ca) {
                World w = w0;
                w.max_new = 40;
                w.cancel_at = ca;
                ws.push_back(w);
            }
            for (int e = 0; e < 8; ++e) {
                World w = w0;
                w.max_new = 40;
                w.eos_pos = w.p0 + 1 + e;
                ws.push_back(w);
            }
            for (int ce = 1; ce <= 10; ++ce) {
                World w = w0;
                w.max_new = 40;
                w.max_context = w.p0 + ce;
                ws.push_back(w);
            }
            for (int cfg = 0; cfg < 2; ++cfg)
                for (size_t vi = 0; vi < ws.size(); ++vi) {
                    const World& w = ws[vi];
                    for (int tmi = 0; tmi < 2; ++tmi) {
                        Timing tm;
                        tm.seed = (uint64_t) pat * 131 + vi * 7 + (uint64_t) tmi;
                        tm.jitter = tmi == 0 ? 0.0 : 0.3;
                        if (tmi == 1) {
                            tm.chain_step *= 2.5;
                            tm.chain_round *= 2.5;
                        }
                        for (int m = 0; m < 3; ++m) {
                            Config c = base_config(kModes[m]);
                            c.max_context = w.max_context;
                            if (cfg == 1) {
                                c.force_miss = 2;
                                c.force_miss_c = 2;
                                c.theta = 0.0f;
                            }
                            const RunResult r = run_model(w, c, tm, false, false);
                            tc[m].add(r);
                            all[m].add(r);
                            if (!r.ok && tc[m].fails <= 3)
                                std::printf("    small #%d/%d/%zu/%d %s: %s\n", pat, cfg, vi, tmi, mode_name(kModes[m]),
                                            r.err.c_str());
                        }
                        if ((pat % 7) == 0 || vi >= 12) {
                            Config c = base_config(Mode::N2);
                            c.max_context = w.max_context;
                            if (cfg == 1) {
                                c.force_miss = 2;
                                c.theta = 0.0f;
                            }
                            compare_n2_old(w, c, tm, cmp, "exhaustive");
                        }
                    }
                }
        }
        for (int m = 0; m < 3; ++m) {
            print_tally("exhaustive", kModes[m], tc[m]);
            check(tc[m].fails == 0, std::string("(c) exhaustive, mode ") + mode_name(kModes[m]));
        }
        std::printf("  [%.1f s]\n", secs());
    }

    // ---- (d) N2 against the old loop (on the worlds of (a), the first 3000 of (b), and part of (c))
    std::printf("(d) N2 against the old loop: %lld worlds, %lld the same line for line, %lld differ\n",
                (long long) cmp.runs, (long long) cmp.same, (long long) cmp.diff);
    if (!cmp.first.empty()) std::printf("  first difference: %s", cmp.first.c_str());
    check(cmp.diff == 0 && cmp.same > 0, "(d) N2 is the old loop");

    // ---- coverage of the verdict cases, per mode (over every input)
    if (only.empty()) {
        std::printf("coverage (every input):\n");
        const int need_d2[] = {kCase0, kCase1c, kCase2, kCase3, kCase4, kCase7, kCase8, kCase9, kCase10};
        const int need_d3[] = {kCase0, kCase1a, kCase1b, kCase1c, kCase1d, kCase2, kCase3, kCase4, kCase5, kCase6,
                               kCase7, kCase8, kCase9, kCase10};
        for (int m = 0; m < 3; ++m) {
            print_tally("all", kModes[m], all[m]);
            const Stats& s = all[m].st;
            if (kModes[m] == Mode::B) {
                for (int c : need_d3) check(s.case_hits[c] > 0, std::string("mode B never hit case ") + case_name(c));
            } else {
                for (int c : need_d2)
                    check(s.case_hits[c] > 0,
                          std::string("mode ") + mode_name(kModes[m]) + " never hit case " + case_name(c));
            }
            check(s.case_hits[kCase11] == 0, "case 11 happened");
            if (kModes[m] != Mode::N2) check(s.restage_spec == 0, std::string("mode ") + mode_name(kModes[m]) +
                                                                    ": a guess window restaged at its launch");
        }
    }
    std::printf("%s (%lld failures) [%.1f s]\n", g_fail == 0 ? "PASS" : "FAIL", (long long) g_fail, secs());
    return g_fail == 0 ? 0 : 1;
}
