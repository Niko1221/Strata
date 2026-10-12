// src/net/stage_link_test.cpp - the remote stage's link over loopback, no GPU: a worker (serve_stage) with handlers
// that copy rows and record the image positions they get, and a StageClient driving it.  What it checks
// (docs/remote-stage/ENGINEERING.md, "Images"):
//
//   0. the worker's table from what a Positions message carries (stage_positions_fill): the identity, the cells
//      sent, the cells past them;
//   1. the hello: an older side (protocol 4) is refused for its protocol, a main process with --vision refuses a
//      worker without it, a matching one is served;
//   2. Positions carries the table whole (cells x (t, h, w), values past 16 bits) and the identity as no table;
//   3. a failed Positions comes back on the next reply - also when that is a Reset, which drops a failed Commit's
//      error but not the next request's positions - and the link stays usable; a later table replaces a failed one;
//      a handler's success is its return value; a malformed table is the next reply's error, not a reply of its own;
//   4. a worker without a positions handler (started without --vision) answers a Positions with an error naming
//      --vision.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
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
StageHello hello_of(int32_t vision = 1) {
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
    h.vision = vision;
    return h;
}
bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// the worker's side the test looks at
struct Shared {
    std::mutex mu;
    int calls = 0;
    bool identity = false;
    std::vector<int32_t> table;
    bool last(int n, bool ident, const std::vector<int32_t>& t) {
        std::lock_guard<std::mutex> lk(mu);
        return calls == n && identity == ident && table == t;
    }
};

// one worker on a free loopback port, until stop()
struct Worker {
    std::atomic<bool> stop_flag{false};
    std::atomic<int> served{-1};
    StageStats st;
    std::thread t;
    std::string where;
    bool start(const StageHandlers& h, const StageBuffers& buf) {
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            served = -1;
            t = std::thread([&, port] { served = serve_stage("127.0.0.1", port, "", h, buf, st, &stop_flag); });
            const std::string at = "127.0.0.1:" + std::to_string(port);
            // until it listens: the worker's own refusal of a protocol 4 hello (not the system's "connection refused")
            for (int i = 0; i < 100 && served.load() < 0; ++i) {
                StageHello bad = hello_of();
                bad.protocol = 4;
                StageHello peer;
                std::string e;
                StageClient c;
                if (c.connect(at, bad, peer, e) || contains(e, "the worker refused:")) { where = at; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (where.empty()) stop();
        }
        return !where.empty();
    }
    void stop() {
        if (!t.joinable()) return;
        stop_flag = true;
        t.join();
        stop_flag = false;
    }
};
}  // namespace

int main() {
    std::printf("stage_link_test\n");
    // 0. the worker's table from a Positions message (no link)
    {
        std::vector<int32_t> out(10 * 3, -1);
        stage_positions_fill(out.data(), 10, nullptr, 0);
        bool ident = true;
        for (int c = 0; c < 10; ++c) ident = ident && out[c * 3] == c && out[c * 3 + 1] == c && out[c * 3 + 2] == c;
        check(ident, "no table: every cell its own index");
        // as the main process builds it: text 0-1, a 2x2 picture at t = 2 (cells 2-5), the text after it from 4 (the
        // picture's extent, 2, not its 4 cells); 8 cells sent of 10
        const int32_t sent[8 * 3] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5};
        stage_positions_fill(out.data(), 10, sent, 8);
        check(std::equal(sent, sent + 24, out.begin()), "the cells sent are taken as they are");
        check(out[8 * 3] == 6 && out[8 * 3 + 1] == 6 && out[8 * 3 + 2] == 6 && out[9 * 3] == 7 && out[9 * 3 + 2] == 7,
              "the cells past them continue from the last one, as the main process's do");
        const int32_t high[3] = {INT32_MAX - 1, 0, 0};
        stage_positions_fill(out.data(), 3, high, 1);
        check(out[3] == INT32_MAX, "the continuation is computed in 64 bits");
    }
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
    std::atomic<int32_t> worker_vision{1};
    std::atomic<bool> fail_positions{false}, fail_commit{false}, stale_text{false};
    StageHandlers h;
    h.hello = [&](const StageHello&, StageHello& mine, std::string&) {
        mine = hello_of(worker_vision.load());
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
    h.reset = [](std::string&) { return true; };
    h.positions = [&](const int32_t* table, int64_t cells, std::string& e) {
        std::lock_guard<std::mutex> lk(w.mu);
        ++w.calls;
        w.identity = table == nullptr;
        w.table.clear();
        if (table != nullptr) w.table.assign(table, table + cells * 3);
        if (stale_text.load()) e = "a message left from a retry (test)";   // and it succeeds
        // fail_positions, or a table starting at t = -7 (no reply tells when the worker has read a Positions, so a
        // flag flipped right after one is a race; the table's own content is not)
        if (!fail_positions.load() && (table == nullptr || table[0] != -7)) return true;
        e = "the positions failed (test)";
        return false;
    };
    Worker wk;
    if (!wk.start(h, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    StageClient c;
    std::string err;
    StageHello peer;
    auto greet = [&](StageClient& cl, const std::string& at, StageHello mine) {
        err.clear();
        cl.close();
        return cl.connect(at, mine, peer, err);
    };

    // 1. the hello
    StageHello old = hello_of();
    old.protocol = 4;
    check(!greet(c, wk.where, old) && contains(err, "the worker refused:") && contains(err, "protocol 4/"),
          "an older main process (protocol 4) is refused for its protocol");
    worker_vision = 0;
    check(!greet(c, wk.where, hello_of(1)) && contains(err, "--vision"), "a main with --vision refuses a worker without it");
    check(greet(c, wk.where, hello_of(0)), "... one without --vision is served by it");
    worker_vision = 1;
    check(greet(c, wk.where, hello_of(1)) && peer.vision == 1, "both with --vision: served");

    // 2. the table whole, the identity as none
    const int64_t cells = 40;
    std::vector<int32_t> table((size_t) cells * 3);
    for (int64_t i = 0; i < cells; ++i) {   // a grid's (t, h, w), positions past 16 bits
        table[(size_t) i * 3] = 70000;
        table[(size_t) i * 3 + 1] = 70000 + (int32_t) (i / 8);
        table[(size_t) i * 3 + 2] = 70000 + (int32_t) (i % 8);
    }
    err.clear();
    check(c.positions(table.data(), cells, err) && c.reset(err) && w.last(1, false, table),
          "a table arrives whole, before the reset");
    check(c.positions(nullptr, 0, err) && c.reset(err) && w.last(2, true, {}), "the identity arrives as no table");

    // 3. a failed Positions outlives the reset; a failed Commit does not
    const int32_t one[3] = {5, 5, 5};
    fail_positions = true;
    err.clear();
    bool sent = c.positions(one, 1, err);
    check(sent && !c.reset(err) && contains(err, "the positions failed (test)"), "a failed Positions comes back on the reset");
    fail_positions = false;
    err.clear();
    check(c.reset(err), "... once: the link is usable after it");
    fail_commit = true;
    check(c.commit(1, err) && c.positions(one, 1, err) && c.reset(err) && w.last(4, false, {5, 5, 5}),
          "a failed Commit is dropped by the reset (the positions after it applied)");
    fail_positions = true;
    err.clear();
    sent = c.commit(1, err) && c.positions(nullptr, 0, err);
    check(sent && !c.reset(err) && contains(err, "the positions failed (test)") && !contains(err, "a commit failed"),
          "a failed Commit, then a failed Positions: the reset reports the positions only");
    fail_commit = false;
    fail_positions = false;
    const int32_t failing[3] = {-7, -7, -7};
    err.clear();
    sent = c.positions(failing, 1, err);
    check(sent && c.positions(nullptr, 0, err) && c.reset(err) && w.last(7, true, {}),
          "a later table replaces a failed one (a refused request: no reset)");
    stale_text = true;
    check(c.positions(one, 1, err) && c.reset(err), "a handler that succeeds with text left in its error: a success");
    stale_text = false;
    std::vector<int32_t> huge((size_t) (((int64_t) 1 << 22) + 1) * 3, 1);
    err.clear();
    sent = c.positions(huge.data(), (int64_t) (huge.size() / 3), err);
    check(sent && !c.reset(err) && contains(err, "bad position table size") && w.last(8, false, {5, 5, 5}),
          "a malformed table (over 4M cells): the next reply's error, no handler call");
    check(c.reset(err), "... and only that reply's: the link is in step after it");
    float rows[kHbf] = {1, 2, 3, 4}, back[kHbf] = {};
    const int32_t tok = 7;
    fail_positions = true;
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.run(1, &tok, 0, rows, kHbf, back, kHbf, err) && contains(err, "the positions failed (test)"),
          "a failed Positions comes back on the next window too");
    fail_positions = false;
    err.clear();
    check(c.run(1, &tok, 0, rows, kHbf, back, kHbf, err) && std::memcmp(rows, back, sizeof rows) == 0,
          "... and the next window runs");
    c.close();
    wk.stop();

    // 4. a worker started without --vision: no positions handler
    worker_vision = 0;
    StageHandlers plain = h;
    plain.positions = nullptr;
    Worker wk2;
    if (!wk2.start(plain, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet(c, wk2.where, hello_of(0)), "a main without --vision is served by a worker without it");
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.reset(err) && contains(err, "--vision"), "a Positions to it is an error naming --vision");
    c.close();
    wk2.stop();

    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
