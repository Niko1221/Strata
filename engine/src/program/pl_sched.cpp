// The pipelined decode loop's scheduler (see pl_sched.hpp).  "Today" is the pw-2 loop in generate.cpp (the
// --pipeline-windows 2 decode loop), which the N2 mode reproduces operation for operation.
#include "pl_sched.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::pl {

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::N2: return "N2";
        case Mode::B0: return "B0";
        case Mode::B: return "B";
    }
    return "?";
}

const char* op_name(Op o) {
    switch (o) {
        case Op::Gap: return "Gap";
        case Op::Prestage: return "Prestage";
        case Op::SnapTake: return "SnapTake";
        case Op::SnapWait: return "SnapWait";
        case Op::Launch: return "Launch";
        case Op::SrcWait: return "SrcWait";
        case Op::Commit: return "Commit";
        case Op::Restore: return "Restore";
        case Op::PleSet: return "PleSet";
        case Op::ChainWaitIdle: return "ChainWaitIdle";
        case Op::ChainLaunch: return "ChainLaunch";
        case Op::ChainExtend: return "ChainExtend";
        case Op::ChainClose: return "ChainClose";
        case Op::AdaptVerdict: return "AdaptVerdict";
        case Op::AdaptTick: return "AdaptTick";
    }
    return "?";
}

const char* case_name(int c) {
    static const char* n[kCaseN] = {"0 next not launched", "1a held, C launched", "1b held, C made",
                                    "1c held, no C", "1d held, C being made", "2 whole, bonus wrong",
                                    "3 not whole", "4 last", "5 held on a guess", "6 wrong on a guess",
                                    "7 forced chain disagrees", "8 T cut by the chain's end", "9 lookup",
                                    "10 context edge", "11 verdict with an undo pending"};
    return c >= 0 && c < kCaseN ? n[c] : "?";
}

bool Sched::fail(const std::string& m) {
    if (err_.empty()) err_ = m;
    return false;
}

bool Sched::run(PlIo& io, PlOp o) {
    if (!io.exec(o)) return fail(std::string("pipelined decode: ") + op_name(o.op) + " failed");
    return true;
}

void Sched::ple_adv(int32_t ple[2], const int32_t* tok, int n) const {
    for (int i = 0; i < n; ++i) {
        ple[0] = ple[1];
        ple[1] = tok[i];
    }
}

PW* Sched::find(uint64_t uid) {
    if (uid == 0) return nullptr;
    for (int k = 0; k < nwin_; ++k)
        if (win_[k].uid == uid) return &win_[k];
    for (int k = 0; k < ndoom_; ++k)
        if (doom_[k].uid == uid) return &doom_[k];
    return nullptr;
}

bool Sched::is_doomed(uint64_t uid) const {
    for (int k = 0; k < ndoom_; ++k)
        if (doom_[k].uid == uid) return true;
    return false;
}

int Sched::slot_owner_count() const {
    int n = 0;
    for (int q = 0; q < c_.ns(); ++q) n += owner_[q] != 0 ? 1 : 0;
    return n;
}

bool Sched::in_flight() const {
    for (int k = 0; k < nwin_; ++k)
        if (win_[k].fl || (win_[k].s1 && !win_[k].s1_done)) return true;
    for (int k = 0; k < ndoom_; ++k)
        if (doom_[k].fl) return true;
    return false;
}

// the serial loop's window size over the chain's outputs from index `from` (`avail` of them) - today's t_rule
int Sched::t_rule(const float* pr, int from, int avail) const {
    int T = c_.S_mtp;
    if (c_.spec_min_p > 0.0) {
        T = 1;
        while (T < c_.S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) c_.spec_min_p) ++T;
    }
    return std::max(1, std::min(T, 1 + avail - from));
}

bool Sched::begin(const Config& c, int64_t p, int32_t x, const int32_t ple_prev[2], double t0_ms, std::string& err) {
    c_ = c;
    if (c_.n_front < 1 || c_.n_front > kMaxFront) { err = "pl_sched: front stages out of range"; return false; }
    if (c_.ns() < 2 || c_.ns() > kMaxSlots || (c_.mode != Mode::N2 && c_.ns() < c_.depth() + 1)) {
        err = "pl_sched: slots out of range";
        return false;
    }
    if (c_.S < 1 || c_.S > kMaxRows || c_.S_mtp < 1 || c_.S_mtp > c_.S) { err = "pl_sched: window size out of range"; return false; }
    if (c_.chain_out2 < 1 || c_.chain_out2 > kChainMax || c_.chain_max < 1 || c_.chain_max > kChainMax) {
        err = "pl_sched: chain outputs out of range";
        return false;
    }
    err_.clear();
    st_ = Stats{};
    nwin_ = ndoom_ = nheld_ = 0;
    undo_ = UndoJob{};
    ch_ = ChainState{};
    req_ = ChainReq{};
    for (int q = 0; q < kMaxSlots; ++q) {
        owner_[q] = 0;
        src_armed_[q] = false;
    }
    ending_ = false;
    next_uid_ = 1;
    prev_held_ = false;
    last_verdict_ = t0_ms;
    end_ = p;
    PW& A = win_[0];
    A = PW{};
    A.T = 1;
    A.p = p;
    A.tok[0] = x;
    A.ready = true;
    A.ple_in[0] = ple_prev[0];
    A.ple_in[1] = ple_prev[1];
    A.uid = next_uid_++;
    nwin_ = 1;
    return true;
}

// ---- the slot pool
int Sched::alloc_slot(PlIo& io, PW& w, bool& waited) {
    waited = false;
    if (w.slot >= 0) return w.slot;
    const int ns = c_.ns();
    int q = w.seq % ns;
    if (owner_[q] != 0) {
        if (c_.mode == Mode::N2) {   // today's parity: never taken (the old loop's !doomed rules keep it so)
            fail("pl_sched N2: slot seq & 1 is taken");
            return -1;
        }
        q = -1;
        for (int i = 0; i < ns; ++i)
            if (owner_[i] == 0) { q = i; break; }
    }
    if (q < 0) {
        waited = true;
        if (w.t_slot_wait < 0.0) {
            w.t_slot_wait = io.now_ms();
            ++st_.slot_waits;
            io.trace("SW", w.seq, nwin_, 0);
        }
        return -1;
    }
    if (w.t_slot_wait >= 0.0) {
        st_.slot_wait_ms += io.now_ms() - w.t_slot_wait;
        w.t_slot_wait = -1.0;
    }
    owner_[q] = w.uid;
    w.slot = q;
    return q;
}

void Sched::free_slot(int q, uint64_t uid) {
    if (q < 0) return;
    if (owner_[q] != uid) {
        fail("pl_sched: a slot given back by a window that does not hold it");
        return;
    }
    owner_[q] = 0;
}

// ---- 1. a window in flight on a front stage served
bool Sched::pump(PlIo& io, PW& w, bool doomed) {
    if (!w.fl) return true;
    const int r = io.front_poll(w.fr, w.slot, doomed);
    if (r < 0) return fail("pipelined decode: a front stage failed");
    if (r == 1) {
        w.fl = false;
        if (++w.fr == c_.n_front) w.finished = true;
        io.trace("F0", w.seq, w.fr, w.spec ? 1 : 0);
    }
    return true;
}

// ---- 2. the undo (sections 3.3, 2.2): N2 once every doomed window has stopped, all stages at once (today); B0/B
// per stage, as soon as no doomed window runs on it.  On each stage: [the doomed windows' graphs] -> restore ->
// replay -> [the fresh window]: the stream's order is today's, only the stages no longer wait for each other.
bool Sched::undo_round(PlIo& io) {
    if (!undo_.active) return true;
    auto stage = [&](int st) -> bool {
        PlOp o;
        o.op = Op::Restore;
        o.st = st;
        o.slot = undo_.slot;
        o.seq = undo_.seq;
        o.uid = undo_.uid;
        if (!run(io, o)) return false;
        if (st == 0) {
            PlOp q;
            q.op = Op::PleSet;
            q.st = 0;
            q.slot = undo_.slot;
            q.seq = undo_.seq;
            q.uid = undo_.uid;
            q.ple[0] = undo_.ple_in[0];
            q.ple[1] = undo_.ple_in[1];
            if (!run(io, q)) return false;
        }
        PlOp m;
        m.op = Op::Commit;
        m.st = st;
        m.slot = undo_.slot;
        m.seq = undo_.seq;
        m.uid = undo_.uid;
        m.keep = undo_.keep;
        m.p = undo_.p;
        return run(io, m);
    };
    if (!c_.per_stage()) {
        for (int d = 0; d < ndoom_; ++d)
            if (doom_[d].fl) return true;
        for (int st = 0; st < c_.n_front; ++st)
            if ((undo_.pending & (1u << st)) != 0 && !stage(st)) return false;
        undo_.pending = 0;
    } else {
        for (int st = 0; st < c_.n_front; ++st) {
            if ((undo_.pending & (1u << st)) == 0) continue;
            bool busy = false;
            for (int d = 0; d < ndoom_; ++d) busy = busy || (doom_[d].fl && doom_[d].fr == st);
            if (busy) continue;
            const double now = io.now_ms();
            if (now - undo_.t0 > 0.05) {   // waited for a doomed window to stop here
                ++st_.stage_waits;
                st_.stage_wait_ms += now - undo_.t0;
            }
            if (!stage(st)) return false;
            undo_.pending &= ~(1u << st);
            io.trace("US", undo_.seq, st, 0);
        }
    }
    if (undo_.pending != 0) return true;
    undo_.active = false;
    ++st_.undo;
    io.trace("U", undo_.seq, undo_.keep, c_.per_stage() ? undo_.n_doomed : 0);
    for (int h = 0; h < nheld_; ++h)
        if (held_[h].uid == undo_.uid) held_[h].need_undo = false;
    return true;
}

// ---- 3. slots back: a doomed window that has stopped; a verified window whose commits and chain are queued
void Sched::release() {
    int n = 0;
    for (int d = 0; d < ndoom_; ++d) {
        if (!doom_[d].fl) {
            free_slot(doom_[d].slot, doom_[d].uid);
            continue;
        }
        if (n != d) doom_[n] = doom_[d];
        ++n;
    }
    ndoom_ = n;
    n = 0;
    for (int h = 0; h < nheld_; ++h) {
        if (!held_[h].need_undo && !held_[h].need_chain) {
            free_slot(held_[h].slot, held_[h].uid);
            continue;
        }
        if (n != h) held_[n] = held_[h];
        ++n;
    }
    nheld_ = n;
}

// ---- the prompt lookup for a fresh window (today's pick_lookup): where the text repeats an earlier stretch and the
// drafter's own first guess agrees, the policy may take the lookup's continuation (up to S rows); its continuation
// past the window is the next window (a lookup window gets no chain-made window behind it)
bool Sched::pick_lookup(PlIo& io, PW& w, const int32_t* chain0) {
    if (!c_.suffix || w.seq == 0) return true;
    Lookup L;
    const int k = io.lookup_propose(2 * c_.S - 1, L);
    if (k <= 0 || L.tok[0] != chain0[0]) return true;
    int t = 0;
    if (!io.lookup_choose(w.T, std::min(k, c_.S - 1), L.match, t)) return true;
    if (t < 1 || t > c_.S) return fail("pl_sched: the lookup's window size is out of range");
    w.T = t;
    for (int i = 1; i < w.T; ++i) {
        w.tok[i] = L.tok[i - 1];
        w.prob[i - 1] = 1.0f;
    }
    w.sfx = true;
    w.sfx_match = L.match;
    const int rest = k - (w.T - 1);   // lookup tokens past w: the next window's row 0 (w's bonus guess) and its drafts
    if (rest >= 1 && c_.lookup_pon > 0.0f && ch_.targets > 0 && nwin_ == 1 && &w == &win_[0]) {
        PW b;
        b.seq = w.seq + 1;
        b.p = w.p + w.T;
        b.T = std::min(c_.S, rest);
        for (int i = 0; i < b.T; ++i) b.tok[i] = L.tok[w.T - 1 + i];
        for (int i = 1; i < b.T; ++i) b.prob[i - 1] = 1.0f;
        b.p_on = c_.lookup_pon;
        b.ready = true;
        b.made = true;
        b.lk = true;
        b.lvl = 1;
        b.uid = next_uid_++;
        b.ple_in[0] = w.ple_in[0];
        b.ple_in[1] = w.ple_in[1];
        ple_adv(b.ple_in, w.tok, w.T);
        win_[nwin_++] = b;
        ch_.targets = 0;   // the chain in flight makes no window for this one
        io.trace("CD", w.seq + 1, 3, (int) (1000.0f * b.p_on));
    }
    return true;
}

// a fresh window's size from the chain's first outputs (today: A once a draft below spec_min_p ends it)
bool Sched::decide_fresh(PlIo& io, PW& A, const int32_t* oc, const float* op, int avail, bool early) {
    A.T = t_rule(op, 0, avail);
    for (int i = 1; i < A.T; ++i) {
        A.tok[i] = oc[i - 1];
        A.prob[i - 1] = op[i - 1];
    }
    if (!pick_lookup(io, A, oc)) return false;
    A.ready = true;
    ch_.early_used = true;
    if (early) io.trace("CE", A.seq, A.T, 0);
    ch_.pred_uid = A.uid;
    ch_.base = A.T - 1;
    ch_.pa_ready = false;
    if (c_.per_stage() && !try_prestage(io, 0)) return false;
    return true;
}

// the next new window from the chain's outputs (today's make_b): its row 0 is the guess at index `base`, its drafts
// follow.  true: decided (made or not); false: not decided yet (or an error, err_ set)
bool Sched::decide_next(PlIo& io, int k, bool done) {
    PW* pred = nwin_ > 0 ? find(ch_.pred_uid) : nullptr;
    if (pred == nullptr || pred != &win_[nwin_ - 1]) {   // the window it follows has been dropped
        ch_.targets = 0;
        return false;
    }
    const int base = ch_.base;
    const int32_t* oc = io.chain_tok();
    const float* op = io.chain_prob();
    bool decided = done || k >= base + c_.S_mtp;
    if (!decided && k > base + 1 && c_.spec_min_p > 0.0)
        for (int j = base + 1; j < k; ++j)
            if (op[j] < (float) c_.spec_min_p) { decided = true; break; }
    if (!decided) return false;
    if (done && k < base + c_.S_mtp) ++st_.case_hits[kCase8];   // the chain ended before the window's size said so
    float pa = 1.0f;
    if (ch_.pa_ready) {
        pa = ch_.pa;
    } else if (ch_.kind == 1) {
        // forced through the windows in flight: its outputs there are the drafter's own picks from the verified
        // window, a better estimate of their acceptance than the chain that drafted them; a disagreement means they
        // are unlikely to be accepted whole
        bool agree = true;
        for (int i = 0; i < ch_.n_force; ++i) {
            if (oc[i] != ch_.force[i]) agree = false;
            pa *= c_.agree ? op[i] : ch_.force_prob[i];
        }
        if (c_.agree && !agree) {
            pa = 0.0f;
            ++st_.disagree;
            ++st_.case_hits[kCase7];
        }
    } else {
        for (int i = 0; i + 1 < pred->T; ++i) pa *= pred->prob[i];
    }
    const int lvl = nwin_;
    bool make = base < k && !pred->sfx && nwin_ < c_.depth();
    if (make && lvl >= 2)   // v1: no window two guesses deep behind a lookup window (case 9)
        for (int j = 0; j < nwin_; ++j) make = make && !win_[j].sfx && !win_[j].lk;
    PW* Y = nullptr;
    if (make) {
        PW y;
        y.seq = pred->seq + 1;
        y.p = pred->p + pred->T;
        y.tok[0] = oc[base];
        y.T = t_rule(op, base + 1, k);
        for (int i = 1; i < y.T; ++i) {
            y.tok[i] = oc[base + i];
            y.prob[i - 1] = op[base + i];
        }
        y.p_on = pa * op[base];
        const int fm = lvl >= 2 ? c_.force_miss_c : c_.force_miss;
        if (fm > 0 && ++(lvl >= 2 ? st_.fm_c : st_.fm) % fm == 0) y.tok[0] = y.tok[0] == 0 ? 1 : 0;
        y.ready = true;
        y.made = true;
        y.lvl = (uint8_t) lvl;
        y.uid = next_uid_++;
        y.ple_in[0] = pred->ple_in[0];
        y.ple_in[1] = pred->ple_in[1];
        ple_adv(y.ple_in, pred->tok, pred->T);
        win_[nwin_++] = y;
        Y = &win_[nwin_ - 1];
        pred = &win_[nwin_ - 2];
        if (lvl >= 2) ++st_.c_made;
    }
    if (c_.mode == Mode::N2) {
        // today: stage it now (its PLE rows from the tokens before it as they will be once its predecessor is
        // committed whole), so its launch is only the graph launch - never while a rollback is pending
        if (Y != nullptr && Y->ready && Y->p_on >= c_.theta && c_.prestage && !undo_.active) {
            int32_t prev[2];
            io.ple_prev(prev);
            ple_adv(prev, pred->tok, pred->T);
            if (prev[0] != Y->ple_in[0] || prev[1] != Y->ple_in[1])
                return fail("pl_sched N2: today's PLE prev and the ple_in chain differ");
            bool waited = false;
            if (alloc_slot(io, *Y, waited) < 0) return fail(err_.empty() ? "pl_sched N2: no slot to prestage" : err_);
            PlOp o;
            o.op = Op::Prestage;
            o.st = 0;
            o.slot = Y->slot;
            o.seq = Y->seq;
            o.uid = Y->uid;
            o.T = Y->T;
            o.tok = Y->tok;
            o.p = Y->p;
            o.ple[0] = prev[0];
            o.ple[1] = prev[1];
            if (!run(io, o)) return false;
            Y->prestaged = true;
        }
    } else if (Y != nullptr && !try_prestage(io, nwin_ - 1)) {
        return false;
    }
    io.trace(lvl >= 2 ? "CD2" : "CD", pred->seq, ch_.kind, (int) (1000.0f * (Y != nullptr ? Y->p_on : 0.0f)));
    --ch_.targets;
    ch_.done_first = true;
    if (Y != nullptr) {
        ch_.pred_uid = Y->uid;
        ch_.base = base + Y->T;
        ch_.pa = Y->p_on;
        for (int i = 0; i + 1 < Y->T; ++i) ch_.pa *= Y->prob[i];
        ch_.pa_ready = true;
    } else {
        ch_.targets = 0;
    }
    return true;
}

// ---- 5. the drafter's chain, read as its outputs land: a fresh window once its size is decided, then each new
// window once ITS size is decided (its row 0 and a draft below spec_min_p, or S_mtp - 1 drafts)
bool Sched::chain_round(PlIo& io) {
    if (ch_.kind == 0) return true;
    const int k_ready = io.chain_ready();
    if (k_ready < 0) return fail("pipelined decode: the drafter's chain failed");
    if (ch_.cancelled) {   // a miss (or the end) made its windows moot: only its end is waited for
        const int r = io.chain_poll();
        if (r < 0) return fail("pipelined decode: the drafter's chain failed");
        if (r == 1) ch_.kind = 0;
        return true;
    }
    // kind 2: the fresh window it sizes (B0/B: once sized, it may be verified while the chain still makes the
    // windows behind it)
    PW* A = ch_.kind == 2 && !ch_.early_used ? find(ch_.a_uid) : nullptr;
    if (ch_.kind == 2 && !ch_.early_used && A == nullptr) return fail("pl_sched: the fresh window of the chain is gone");
    if (ch_.kind == 2 && !ch_.early_used) {
        const int k = k_ready;
        bool decided = k >= c_.S_mtp - 1;
        if (!decided && c_.spec_min_p > 0.0)
            for (int j = 0; j < k; ++j)
                if (io.chain_prob()[j] < (float) c_.spec_min_p) { decided = true; break; }
        if (decided && !decide_fresh(io, *A, io.chain_tok(), io.chain_prob(), std::min(k, c_.S_mtp - 1), true))
            return false;
    }
    const int r = io.chain_poll();
    if (r < 0) return fail("pipelined decode: the drafter's chain failed");
    const int k = r == 1 ? ch_.n : k_ready;
    if (r == 1 && ch_.kind == 2 && !ch_.early_used) {
        if (A == nullptr) return fail("pl_sched: the fresh window of the chain is gone");
        if (!decide_fresh(io, *A, io.chain_tok(), io.chain_prob(), std::min(ch_.n, c_.S_mtp - 1), false)) return false;
    }
    while (ch_.targets > 0 && (ch_.kind == 1 || ch_.early_used))
        if (!decide_next(io, k, r == 1)) break;
    if (!err_.empty()) return false;
    if (ch_.open && r == 0) {
        if (ch_.targets == 0) {   // every new window decided: no more steps
            PlOp o;
            o.op = Op::ChainClose;
            if (!run(io, o)) return false;
            ch_.open = false;
        } else if (k_ready + 1 > ch_.asked) {   // one step ahead of what has landed
            PlOp o;
            o.op = Op::ChainExtend;
            o.k = k_ready + 1;
            if (!run(io, o)) return false;
            ch_.asked = k_ready + 1;
            const int q = std::min(ch_.n, ch_.asked + 1);
            if (q > ch_.queued) {
                st_.chain_steps += q - ch_.queued;
                ch_.queued = q;
            }
            if (ch_.queued >= ch_.n) ch_.open = false;
            io.trace("CX", ch_.src_seq, ch_.asked, 0);
        }
    }
    if (r == 1) ch_.kind = 0;
    return true;
}

// a chain: N2 at the verdict (today), B0/B when the request's turn comes
bool Sched::launch_chain(PlIo& io, const ChainReq& r) {
    PlOp o;
    o.op = Op::ChainLaunch;
    o.slot = r.src_slot;
    o.uid = r.src_uid;
    o.seq = r.src_seq;
    o.T = r.T;
    o.outp = r.outp;
    o.p = r.p;
    o.a = r.a;
    o.kind = r.kind;
    ch_ = ChainState{};
    ch_.kind = r.kind;
    ch_.src_seq = r.src_seq;
    const int depth = c_.depth();
    const int cap = depth >= 3 ? c_.chain_max : c_.chain_out2;
    int n_out = 0;
    if (r.kind == 2) {
        if (nwin_ < 1 || win_[0].uid != r.a_uid) return fail("pl_sched: the fresh window of a chain is gone");
        ch_.a_uid = r.a_uid;
        ch_.targets = depth - 1;
        n_out = cap;
    } else {
        // the windows behind the source: N2 launches the chain before the verified window leaves win_ (today's
        // order), B0/B after
        const int f = nwin_ > 0 && win_[0].uid == r.src_uid ? 1 : 0;
        if (nwin_ < f + 1 || win_[f].seq != r.src_seq + 1 || !win_[f].launched)
            return fail("pl_sched: a forced chain without the window behind its source");
        const PW& X1 = win_[f];
        for (int i = 1; i < X1.T; ++i) {
            ch_.force[ch_.n_force] = X1.tok[i];
            ch_.force_prob[ch_.n_force++] = X1.prob[i - 1];
        }
        if (depth >= 3 && nwin_ >= f + 2) {   // through the window behind it too (C): the chain then makes D
            const PW& X2 = win_[f + 1];
            for (int i = 0; i < X2.T; ++i) {
                ch_.force[ch_.n_force] = X2.tok[i];
                ch_.force_prob[ch_.n_force++] = i == 0 ? 1.0f : X2.prob[i - 1];
            }
        }
        ch_.targets = depth - (nwin_ - f);
        ch_.pred_uid = win_[nwin_ - 1].uid;
        ch_.base = ch_.n_force;
        ch_.pa_ready = false;
        n_out = std::min(cap, ch_.n_force + c_.S_mtp * std::max(1, ch_.targets));
    }
    for (int i = 0; i < ch_.n_force; ++i) o.force[i] = ch_.force[i];
    o.n_force = ch_.n_force;
    o.n_out = n_out;
    o.n_early = n_out;
    o.n_queue = c_.mode == Mode::B && c_.chain_stop ? std::min(n_out, ch_.n_force + 2) : 0;
    if (o.n_queue >= n_out) o.n_queue = 0;
    if (!run(io, o)) return false;
    ch_.n = io.chain_n();
    if (ch_.n < n_out) ++st_.case_hits[kCase10];
    ch_.open = o.n_queue > 0 && o.n_queue < ch_.n;
    ch_.asked = o.n_queue - 1;
    ch_.queued = o.n_queue > 0 ? std::min(o.n_queue, ch_.n) : ch_.n;
    st_.chain_steps += ch_.queued;
    ++st_.chain_launches;
    if (c_.per_stage() && r.src_slot >= 0) src_armed_[r.src_slot] = true;
    for (int h = 0; h < nheld_; ++h)
        if (held_[h].uid == r.src_uid) held_[h].need_chain = false;
    io.trace("CL", r.src_seq, r.kind, 0);
    return true;
}

// ---- 6. (B0/B) the chain asked for at a verdict, once the drafter is idle, assembled from the windows known now
bool Sched::request_round(PlIo& io) {
    if (!req_.pending || ch_.kind != 0 || io.chain_live()) return true;
    if (req_.waited) st_.chain_wait_ms += io.now_ms() - req_.t0;
    req_.pending = false;
    return launch_chain(io, req_);
}

// (B0/B) a window staged ahead of its launch on stage 0: its PLE rows from ple_in, the tokens before it as they will
// be when it launches there, whatever the stages hold now
bool Sched::try_prestage(PlIo& io, int k) {
    PW& w = win_[k];
    if (!c_.prestage || !w.ready || w.launched || w.prestaged || w.gated || w.fl) return true;
    if (k >= 1 && w.p_on < c_.gate(k)) return true;   // below its gate: it will not launch
    if (k >= 1 && w.p + w.T > c_.max_context) return true;
    bool waited = false;
    if (alloc_slot(io, w, waited) < 0) return err_.empty();
    PlOp o;
    o.op = Op::Prestage;
    o.st = 0;
    o.slot = w.slot;
    o.seq = w.seq;
    o.uid = w.uid;
    o.T = w.T;
    o.tok = w.tok;
    o.p = w.p;
    o.ple[0] = w.ple_in[0];
    o.ple[1] = w.ple_in[1];
    if (!run(io, o)) return false;
    w.prestaged = true;
    return true;
}

void Sched::drop_from(int k) {
    for (int j = k; j < nwin_; ++j) {
        if (win_[j].slot >= 0) free_slot(win_[j].slot, win_[j].uid);
    }
    nwin_ = std::max(0, std::min(nwin_, k));
    if (ch_.kind != 0 && ch_.targets > 0 && find(ch_.pred_uid) == nullptr) ch_.targets = 0;
}

// ---- 7. the front stages: each window on its next stage, oldest first (every stage's stream gets its windows in
// order).  The oldest when the stage's state is its own (no undo pending there); one behind it one stage behind
// the window before, which is committed there speculatively (its snapshot taken first) right before it launches.
bool Sched::launches(PlIo& io, bool& stop) {
    stop = false;
    for (int k = 0; k < nwin_; ++k) {
        PW& w = win_[k];
        if (!w.ready || w.fl || w.fr >= c_.n_front || w.gated) continue;
        const int st = w.fr;
        const uint32_t bit = 1u << st;
        if (c_.per_stage() ? (undo_.active && (undo_.pending & bit) != 0) : undo_.active) continue;
        if (k >= 1) {
            const PW& v = win_[k - 1];
            if (w.fr >= c_.spec_depth || !(v.fr > st)) continue;
            if (st == 0 && !(w.p_on >= c_.gate(k) && w.p + w.T <= c_.max_context)) {
                ++st_.gate;
                if (w.p + w.T > c_.max_context) ++st_.case_hits[kCase10];
                if (c_.mode == Mode::N2) {   // today: B.ready = false (A's verdict decides the next window)
                    w.ready = false;
                    w.gated = true;
                } else {
                    drop_from(k);
                }
                break;
            }
        } else if (st == 0 && w.p + w.T > c_.max_context) {
            ++st_.case_hits[kCase10];
            ending_ = true;
            stop = true;
            return true;
        }
        bool waited = false;
        if (alloc_slot(io, w, waited) < 0) {
            if (!err_.empty()) return false;
            continue;
        }
        if (st == 0) {
            PlOp g;
            g.op = Op::Gap;
            g.st = 0;
            if (!run(io, g)) return false;
        }
        if (k >= 1) {
            PW& v = win_[k - 1];
            PlOp sw;
            sw.op = Op::SnapWait;
            sw.st = st;
            sw.slot = v.slot;
            sw.seq = v.seq;
            sw.uid = v.uid;
            if (!run(io, sw)) return false;
            if (st == 0 && c_.mode == Mode::N2) {
                io.ple_prev(undo_ple_);
                if (undo_ple_[0] != v.ple_in[0] || undo_ple_[1] != v.ple_in[1])
                    return fail("pl_sched N2: today's undo_ple and the window's ple_in differ");
            }
            PlOp cm;
            cm.op = Op::Commit;
            cm.st = st;
            cm.slot = v.slot;
            cm.seq = v.seq;
            cm.uid = v.uid;
            cm.keep = v.T;
            cm.p = v.p;
            cm.spec = true;
            if (!run(io, cm)) return false;
            v.com |= bit;
        }
        if (st == 0) {   // ss.ple_prev must be the window's ple_in now (the prestage used it)
            int32_t cur[2];
            io.ple_prev(cur);
            if (cur[0] != w.ple_in[0] || cur[1] != w.ple_in[1])
                return fail("pl_sched: ss.ple_prev is not the window's ple_in at its launch");
            if (!w.prestaged) {
                ++st_.restage;
                if (k >= 1 && c_.prestage) ++st_.restage_spec;   // a guess that could have been staged ahead
            }
            w.prestaged = false;
        }
        PlOp sn;
        sn.op = Op::SnapTake;
        sn.st = st;
        sn.slot = w.slot;
        sn.seq = w.seq;
        sn.uid = w.uid;
        if (!run(io, sn)) return false;
        PlOp l;
        l.op = Op::Launch;
        l.st = st;
        l.slot = w.slot;
        l.seq = w.seq;
        l.uid = w.uid;
        l.T = w.T;
        l.tok = w.tok;
        l.p = w.p;
        if (!run(io, l)) return false;
        w.fl = true;
        if (st == 0) {
            w.launched = true;
            if (k >= 1) {
                w.spec = true;
                ++st_.spec;
                if (k >= 2) ++st_.c_launched;
            }
        }
        io.trace("L0", w.seq, w.T, 10 * k + st);
    }
    return true;
}

// ---- 8. the last stage: the oldest window, once every front stage is done
bool Sched::last_stage(PlIo& io) {
    if (nwin_ == 0) return true;
    PW& w = win_[0];
    if (!w.finished || w.s1) return true;
    if (ch_.kind == 1 && !(nwin_ >= 2 && win_[1].ready)) ++st_.late;   // stage 0 idles until the chain has the next
    if (c_.per_stage() && src_armed_[w.slot]) {   // the chain read from this slot's last rows has copied them
        PlOp s;
        s.op = Op::SrcWait;
        s.st = c_.n_front;
        s.slot = w.slot;
        s.seq = w.seq;
        s.uid = w.uid;
        if (!run(io, s)) return false;
        src_armed_[w.slot] = false;
    }
    PlOp g;
    g.op = Op::Gap;
    g.st = c_.n_front;
    if (!run(io, g)) return false;
    PlOp l;
    l.op = Op::Launch;
    l.st = c_.n_front;
    l.slot = w.slot;
    l.seq = w.seq;
    l.uid = w.uid;
    l.T = w.T;
    l.tok = w.tok;
    l.p = w.p;
    if (!run(io, l)) return false;
    w.s1 = true;
    io.trace("L1", w.seq, w.T, 0);
    return true;
}

// ---- 9. the oldest window's verdict
bool Sched::verdict(PlIo& io) {
    if (nwin_ == 0 || !win_[0].s1_done) return true;
    if (c_.mode == Mode::N2) {
        // today: the drafter must be idle before the next chain (bounded: the chain needs nothing from the host)
        if (ch_.kind != 0 && io.chain_live()) {
            PlOp w;
            w.op = Op::ChainWaitIdle;
            if (!run(io, w)) return false;
        }
        ch_.kind = 0;
    } else if (req_.pending) {   // the chain asked for at the last verdict never started: this verdict asks its own
        ++st_.chain_late;
        for (int h = 0; h < nheld_; ++h)
            if (held_[h].uid == req_.src_uid) held_[h].need_chain = false;
        req_.pending = false;
    }
    PW W = win_[0];
    int a = 0;
    while (a < W.T - 1 && W.tok[a + 1] == outp_[a]) ++a;
    {
        PlOp m;
        m.op = Op::Commit;
        m.st = c_.n_front;
        m.slot = W.slot;
        m.seq = W.seq;
        m.uid = W.uid;
        m.keep = a + 1;
        m.p = W.p;
        if (!run(io, m)) return false;
    }
    bool last = false;
    if (!io.verdict(W, a, outp_, last)) return fail("pipelined decode: the verdict failed");
    ++st_.windows;
    end_ = W.p + a + 1;
    PW* X = nwin_ >= 2 ? &win_[1] : nullptr;
    const bool on = X != nullptr && X->launched && a == W.T - 1 && X->tok[0] == outp_[W.T - 1];
    if (X != nullptr && X->made) {   // the gate's calibration: would X have been on the path, by its estimate p_on
        const bool would = a == W.T - 1 && X->tok[0] == outp_[W.T - 1];
        const int bin = std::min(9, std::max(0, (int) (X->p_on * 10.0f)));
        st_.cal_n[bin] += 1;
        st_.cal_on[bin] += would ? 1 : 0;
    }
    io.trace("V", W.seq, a, on ? 1 : (X != nullptr && X->launched ? -1 : 0));
    {
        const double now = io.now_ms();
        const int c = W.spec ? 1 : 0;
        if (W.seq > 0 && !last) io.observe(W, a, now - last_verdict_);
        st_.cls_ms[c] += now - last_verdict_;
        st_.cls_tok[c] += a + 1;
        st_.cls_n[c] += 1;
        if (W.spec) {
            st_.held_n[prev_held_ ? 1 : 0] += 1;
            st_.held_ms[prev_held_ ? 1 : 0] += now - last_verdict_;
        }
        prev_held_ = W.spec;
        last_verdict_ = now;
    }
    // the verdict cases, as counted
    if (last) {
        ++st_.case_hits[kCase4];
    } else if (X == nullptr || !X->launched) {
        ++st_.case_hits[kCase0];
    } else if (on) {
        if (W.spec) ++st_.case_hits[kCase5];
        const PW* X2 = nwin_ >= 3 ? &win_[2] : nullptr;
        if (X2 != nullptr && X2->launched) ++st_.case_hits[kCase1a];
        else if (X2 != nullptr) ++st_.case_hits[kCase1b];
        else if (ch_.kind != 0 && ch_.targets > 0 && ch_.pred_uid == X->uid && c_.depth() >= 3) ++st_.case_hits[kCase1d];
        else ++st_.case_hits[kCase1c];
    } else {
        ++st_.case_hits[a == W.T - 1 ? kCase2 : kCase3];
        if (W.spec) ++st_.case_hits[kCase6];
    }
    if (W.sfx || (X != nullptr && X->lk)) ++st_.case_hits[kCase9];
    if (!last) {
        PlOp o;
        o.op = Op::AdaptVerdict;
        if (!run(io, o)) return false;
    }
    if (nheld_ >= 4) return fail("pl_sched: too many verified windows hold their slots");
    held_[nheld_++] = Held{W.uid, W.slot, false, !last};
    ChainReq rq;
    rq.src_slot = W.slot;
    rq.src_uid = W.uid;
    rq.src_seq = W.seq;
    rq.T = W.T;
    rq.a = a;
    rq.p = W.p;
    for (int i = 0; i < kMaxRows; ++i) rq.outp[i] = outp_[i];
    rq.t0 = io.now_ms();
    auto commits = [&](int keep) -> bool {   // the front stages no window behind it has reached
        for (int st = 0; st < c_.n_front; ++st) {
            if ((W.com & (1u << st)) != 0) continue;
            PlOp m;
            m.op = Op::Commit;
            m.st = st;
            m.slot = W.slot;
            m.seq = W.seq;
            m.uid = W.uid;
            m.keep = keep;
            m.p = W.p;
            if (!run(io, m)) return false;
        }
        return true;
    };
    auto ask = [&](int kind) -> bool {   // the next chain: N2 now, B0/B when the drafter is idle
        rq.kind = kind;
        if (c_.mode == Mode::N2) return launch_chain(io, rq);
        rq.pending = true;
        if (ch_.kind != 0 || io.chain_live()) {
            rq.waited = true;
            ++st_.chain_waits;
            io.trace("CW", W.seq, kind, 0);
        }
        req_ = rq;
        return true;
    };
    if (on && !last) {
        ++st_.on;
        if (X->lvl >= 2) ++st_.c_on;
        if (c_.mode == Mode::N2 && !ask(1)) return false;   // today: the chain first, over W forced through X
        if (!commits(W.T)) return false;   // W accepted whole: X (now the oldest) runs there after it
        for (int k = 1; k < nwin_; ++k) win_[k - 1] = win_[k];
        --nwin_;
        if (c_.mode != Mode::N2 && !ask(1)) return false;
    } else {
        // the front stages no window behind W reached take its real commit now; those they did reach are undone
        // (snapshot restored, the commit replayed) - N2 once they have all stopped, B0/B stage by stage
        if (!commits(a + 1)) return false;
        int nd = 0;
        for (int k = 1; k < nwin_; ++k) {
            if (win_[k].launched) {
                if (ndoom_ >= kMaxWin) return fail("pl_sched: too many doomed windows");
                win_[k].doomed = true;
                doom_[ndoom_++] = win_[k];
                if (win_[k].lvl >= 2) ++st_.c_doomed;
                ++nd;
            } else if (win_[k].slot >= 0) {
                free_slot(win_[k].slot, win_[k].uid);
            }
        }
        if (nd > 0) {
            if (undo_.active) {   // case 11: impossible (a fresh window passes every front stage before its verdict)
                ++st_.case_hits[kCase11];
                return fail("pl_sched: a verdict with an undo still pending");
            }
            if (c_.mode == Mode::N2 && (undo_ple_[0] != W.ple_in[0] || undo_ple_[1] != W.ple_in[1]))
                return fail("pl_sched N2: today's undo_ple and the window's ple_in differ");
            undo_ = UndoJob{};
            undo_.active = true;
            undo_.uid = W.uid;
            undo_.slot = W.slot;
            undo_.seq = W.seq;
            undo_.keep = a + 1;
            undo_.p = W.p;
            undo_.ple_in[0] = W.ple_in[0];
            undo_.ple_in[1] = W.ple_in[1];
            undo_.pending = W.com;
            undo_.n_doomed = nd;
            undo_.t0 = io.now_ms();
            ++st_.undo_jobs;
            held_[nheld_ - 1].need_undo = true;
        } else if (W.com != 0) {
            return fail("pl_sched: committed speculatively with no window launched behind it");
        }
        if (c_.mode != Mode::N2 && ch_.kind != 0 && !ch_.cancelled) {   // the chain's windows are moot now
            ch_.cancelled = true;
            ++st_.chain_cancel;
            if (ch_.open) {
                PlOp o;
                o.op = Op::ChainClose;
                if (!run(io, o)) return false;
                ch_.open = false;
            }
        }
        nwin_ = 0;
        if (!last) {
            PW nA;
            nA.seq = W.seq + 1;
            nA.p = W.p + a + 1;
            nA.tok[0] = outp_[a];
            nA.ple_in[0] = W.ple_in[0];
            nA.ple_in[1] = W.ple_in[1];
            ple_adv(nA.ple_in, W.tok, a + 1);
            nA.uid = next_uid_++;
            win_[0] = nA;
            nwin_ = 1;
            rq.a_uid = nA.uid;
            if (!ask(2)) return false;
        }
    }
    if (!last) {
        PlOp o;
        o.op = Op::AdaptTick;
        if (!run(io, o)) return false;
    }
    if (last) ending_ = true;
    return true;
}

int Sched::step(PlIo& io) {
    if (!err_.empty()) return -1;
    // 1. every window in flight served: the doomed ones, each front stage's, the oldest on the last stage
    for (int d = 0; d < ndoom_; ++d)
        if (!pump(io, doom_[d], true)) return -1;
    for (int k = 0; k < nwin_; ++k)
        if (!pump(io, win_[k], false)) return -1;
    if (nwin_ > 0 && win_[0].s1 && !win_[0].s1_done) {
        const int r = io.last_poll(win_[0].slot, outp_);
        if (r < 0) return fail("pipelined decode: the last stage failed"), -1;
        if (r == 1) {
            win_[0].s1_done = true;
            io.trace("F1", win_[0].seq, win_[0].T, 0);
        }
    }
    // 2. undo; 3. slots back
    if (!undo_round(io)) return -1;
    release();
    if (!err_.empty()) return -1;
    // 4. the end: no undo pending, nothing doomed or in flight
    if (ending_) {
        if (c_.mode == Mode::N2) return undo_.active ? 1 : 0;
        return undo_.active || ndoom_ > 0 || in_flight() ? 1 : 0;
    }
    // 5. the chain's outputs; 6. the next chain, and the windows that could not be staged before
    if (!chain_round(io)) return -1;
    if (c_.per_stage()) {
        if (!request_round(io)) return -1;
        for (int k = 0; k < nwin_; ++k)
            if (!try_prestage(io, k)) return -1;
    }
    // 7. the front stages; 8. the last stage; 9. the verdict
    bool stop = false;
    if (!launches(io, stop)) return -1;
    if (stop) return 1;
    if (!last_stage(io)) return -1;
    if (!verdict(io)) return -1;
    return err_.empty() ? 1 : -1;
}

}  // namespace strata::pl
