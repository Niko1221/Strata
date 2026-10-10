// src/net/stage_link_test.cpp - the remote stage's link over loopback, no GPU: a worker (serve_stage) with handlers
// that copy rows and keep a StageCkptStore, and a StageClient driving it.  What it checks
// (docs/remote-stage/CHECKPOINTS.md):
//
//   1. the hello: an older side (protocol 3) is refused for its protocol, a main process that would have the worker
//      keep more than kStageCkptMax checkpoints is refused, a matching one is served;
//   2. Reset / CkptSave / CkptRestore carry their id, position and keep[] whole (int64 ids beyond 32 bits);
//   3. the worker's position: a checkpoint is saved only where the session is (0 after a reset, a chunk's end, a
//      window's first position plus its accepted count, L after a restore); a failed Commit comes back on the
//      CkptSave, and a CkptRestore drops it (the session is replaced);
//   4. CkptSaveOk a=0 (not stored) leaves the store as it was; a restore of an id the store does not hold is the
//      missing-id error (the main process's fallback), any other restore error is not;
//   5. a keep[] longer than kStageCkptMax is a malformed message; a disconnect clears the store.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
#include "strata/net/stage_ckpt_store.hpp"
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::net;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr int64_t kHbf = 4, kD = 4, kChunk = 8, kMaxT = 4;
StageHello hello_of() {
    StageHello h;
    h.n_embd = 4;
    h.hc = 1;
    h.n_layers = 4;
    h.n_expert = 2;
    h.layer_begin = 2;
    h.max_t = (int32_t) kMaxT;
    h.max_context = 64;
    h.chunk = kChunk;
    h.handoff_floats = kHbf;
    h.pack_hash = 0x5354524d;
    return h;
}
bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// the worker's side the test looks at
struct Shared {
    std::mutex mu;
    StageCkptStore store;
    std::vector<int64_t> last_keep;
    int64_t last_id = -1, last_L = -1;
    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
    bool has(int64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        return store.find(id) != nullptr;
    }
    bool last(int64_t id, int64_t L, const std::vector<int64_t>& keep) {
        std::lock_guard<std::mutex> lk(mu);
        return last_id == id && last_L == L && last_keep == keep;
    }
    bool last_keep_is(const std::vector<int64_t>& keep) {
        std::lock_guard<std::mutex> lk(mu);
        return last_keep == keep;
    }
};
}  // namespace

int main() {
    std::printf("stage_link_test\n");
    // ---- the worker
    std::vector<float> run_in((size_t) (kMaxT * kHbf)), run_out(run_in.size());
    std::vector<float> pf_in0((size_t) (kChunk * kD)), pf_in1(pf_in0.size()), pf_out0(pf_in0.size()),
        pf_out1(pf_in0.size());
    StageBuffers buf;
    buf.run_in = run_in.data();
    buf.run_out = run_out.data();
    buf.run_floats = run_in.size();
    buf.pf_in[0] = pf_in0.data();
    buf.pf_in[1] = pf_in1.data();
    buf.pf_out[0] = pf_out0.data();
    buf.pf_out[1] = pf_out1.data();
    buf.pf_floats = pf_in0.size();
    buf.handoff_floats = kHbf;
    buf.pf_row_floats = kD;
    Shared w;
    std::atomic<bool> refuse_save{false}, fail_commit{false};
    std::atomic<int> disconnects{0};
    StageHandlers h;
    h.hello = [](const StageHello&, StageHello& mine, std::string&) {
        mine = hello_of();
        return true;
    };
    h.run = [&](int T, const int32_t*, int64_t, std::string&) {
        std::memcpy(run_out.data(), run_in.data(), (size_t) (T * kHbf) * 4);
        return true;
    };
    h.commit = [&](int, std::string& e) {
        if (!fail_commit.load()) return true;
        e = "a commit failed (test)";
        return false;
    };
    h.prefill = [&](const int64_t*, int64_t T, int64_t, int64_t, int64_t, const float* in, float* out, std::string&) {
        std::memcpy(out, in, (size_t) (T * kD) * 4);
        return true;
    };
    h.reset = [&](const std::vector<int64_t>& keep, std::string&) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_keep = keep;
        w.store.prune(keep);
        return true;
    };
    h.ckpt_save = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string&) -> int {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_id = id;
        w.last_L = L;
        w.last_keep = keep;
        if (refuse_save.load() || !w.store.admits(id, keep)) return 0;
        StageCkptStore::Part p;
        p.L = L;
        w.store.put(id, std::move(p), keep);
        return 1;
    };
    h.ckpt_restore = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep, std::string& e) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_keep = keep;
        if (id == 99) {   // a relay's forwarded missing id: its own prefix first
            e = std::string("relay: the next worker: remote stage (worker): ") + kStageCkptMissing + "99";
            return false;
        }
        const StageCkptStore::Part* p = w.store.find(id);
        if (p == nullptr) {
            e = kStageCkptMissing + std::to_string(id);
            return false;
        }
        if (p->L != L) {
            e = "checkpoint saved at another position";
            return false;
        }
        w.store.prune(keep);
        return true;
    };
    h.disconnected = [&] {
        {
            std::lock_guard<std::mutex> lk(w.mu);
            w.store.clear();
        }
        ++disconnects;
    };
    std::atomic<bool> stop{false};
    StageStats wst;
    std::atomic<int> served{-1};
    std::thread server;
    std::string where;
    std::mt19937 rng(std::random_device{}());
    StageClient c;
    std::string err;
    auto greet = [&](StageHello mine) -> bool {
        StageHello peer;
        err.clear();
        return c.connect(where, mine, peer, err);
    };
    // a free port: a few tries (another process may hold one)
    for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
        const int port = 20000 + (int) (rng() % 40000);
        served = -1;
        server = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, wst, &stop); });
        const std::string at = "127.0.0.1:" + std::to_string(port);
        // until it listens: the worker's own refusal of a protocol 3 hello (not the system's "connection refused")
        for (int i = 0; i < 100 && served.load() < 0; ++i) {
            StageHello bad = hello_of();
            bad.protocol = 3;
            StageHello peer;
            std::string e;
            c.close();
            if (c.connect(at, bad, peer, e) || contains(e, "the worker refused:")) { where = at; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (where.empty()) {
            stop = true;
            server.join();
            stop = false;
        }
    }
    if (where.empty()) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }

    // 1. the hello
    StageHello old_side = hello_of();
    old_side.protocol = 3;
    check(!greet(old_side) && contains(err, "protocol 3/4"), "a protocol 3 main process is refused for its protocol");
    StageHello greedy = hello_of();
    greedy.ckpt_max = kStageCkptMax + 1;
    check(!greet(greedy) && contains(err, "at most 64"),
          "a main process asking for more than 64 checkpoints is refused");
    StageHello mine = hello_of();
    mine.ckpt_max = kStageCkptMax;
    check(greet(mine), "a matching main process is served");

    // 2-3. a reset, a prompt chunk, a checkpoint at its end, a window and its commit, another checkpoint
    bool stored = false, missing = false;
    check(c.reset({}, err) && w.last_keep_is({}), "a reset that keeps nothing");
    std::vector<int64_t> tok((size_t) kChunk);
    std::vector<float> rows((size_t) (kChunk * kD)), back(rows.size(), 0.0f);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i;
    for (size_t i = 0; i < tok.size(); ++i) tok[i] = (int64_t) i;
    auto chunk = [&] {
        std::fill(back.begin(), back.end(), 0.0f);
        return c.prefill_send(tok.data(), kChunk, 0, 0, rows.data(), (size_t) kD, 0, err) &&
               c.prefill_recv(back.data(), (size_t) kD, kChunk, 0, err) && back == rows;
    };
    check(chunk(), "a prompt chunk's rows come back");
    check(c.reset({}, err) && !c.ckpt_save(7, 3, {7}, stored, err) && contains(err, "session is at 0") &&
              w.size() == 0,
          "after a reset the session is at 0, not at the chunk's end: a checkpoint elsewhere is a state error");
    check(chunk(), "the prompt chunk again");
    const int64_t big = ((int64_t) 1 << 40) + 7;
    check(!c.ckpt_save(big, 5, {big}, stored, err) && contains(err, "session is at 8") && c.connected(),
          "a checkpoint where the session is not is a state error (the link stays)");
    check(c.ckpt_save(big, kChunk, {big}, stored, err) && stored && w.last(big, kChunk, {big}),
          "CkptSave carries id, position and keep whole; stored at a chunk's end");
    std::vector<int32_t> wtok(3, 1);
    std::vector<float> win((size_t) (3 * kHbf), 2.0f), wout(win.size(), 0.0f);
    check(c.run(3, wtok.data(), kChunk, win.data(), win.size(), wout.data(), wout.size(), err) && wout == win &&
              c.commit(2, err),
          "a window and its commit (two of three accepted)");
    check(c.ckpt_save(2, kChunk + 2, {big, 2}, stored, err) && stored && w.size() == 2,
          "a checkpoint at the window's position plus its accepted count");
    refuse_save = true;
    check(c.ckpt_save(3, kChunk + 2, {big, 2, 3}, stored, err) && !stored && w.size() == 2 && !w.has(3),
          "CkptSaveOk a=0: not stored, the store as it was");
    refuse_save = false;

    // 4. restores
    check(c.ckpt_restore(big, kChunk, {big}, missing, err) && !missing && w.size() == 1 && w.has(big),
          "a restore puts the part back and prunes to its keep");
    check(c.ckpt_save(5, kChunk, {big, 5}, stored, err) && stored && w.size() == 2,
          "after a restore the session is at its L: a checkpoint there is stored");
    check(!c.ckpt_restore(2, kChunk + 2, {big, 2}, missing, err) && missing && c.connected(),
          "a restore of an id the store does not hold is the missing-id error");
    check(!c.ckpt_restore(big, kChunk + 1, {big}, missing, err) && !missing,
          "any other restore error is not the missing-id one");
    check(!c.ckpt_restore(99, kChunk, {big}, missing, err) && !missing && contains(err, kStageCkptMissing),
          "a relay's forwarded missing id is not the missing-id error (the state class)");
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err),
          "a window after the restore");
    fail_commit = true;
    check(c.commit(1, err), "a commit that fails on the worker (one way: no reply)");
    check(!c.ckpt_save(4, kChunk + 1, {big, 5, 4}, stored, err) && contains(err, "a commit failed (test)"),
          "the failed commit comes back on the next CkptSave");
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
              c.commit(1, err),
          "another window whose commit fails");
    // (the restore waits for its reply, so the worker has run that commit, failing, before fail_commit goes back)
    check(c.ckpt_restore(big, kChunk, {big, 5}, missing, err), "a restore while the failed commit's error is pending");
    fail_commit = false;
    check(c.ckpt_save(6, kChunk, {big, 5, 6}, stored, err) && stored && w.size() == 3,
          "a restore drops the failed commit: the next CkptSave is stored");

    // 5. a malformed keep, a reset that keeps one, a disconnect
    std::vector<int64_t> too_many((size_t) kStageCkptMax + 1, 1);
    check(!c.reset(too_many, err) && contains(err, "bad checkpoint list"), "a keep[] of more than 64 ids is refused");
    check(c.reset({big}, err) && w.last_keep_is({big}) && w.size() == 1, "a reset keeps the parts it names");
    const int before = disconnects.load();
    c.close();
    for (int i = 0; i < 200 && disconnects.load() == before; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() > before && w.size() == 0, "a disconnect clears the store");
    check(greet(mine) && !c.ckpt_restore(big, kChunk, {big}, missing, err) && missing,
          "the next main process finds none of the last one's parts");
    c.close();

    stop = true;
    server.join();
    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
