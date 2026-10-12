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
//      --vision;
//   5. a new main process waits while the served one sends messages, and takes it over once it has been silent (not
//      closed) for STRATA_REMOTE_TAKEOVER_S (1 s here): B served, A's next message fails as a lost link; one with a
//      wrong token or another model is refused and A stays; when the served one closes, the one waiting is served at
//      once; which errors stage_link_lost counts as a lost link;
//   6. STRATA_REMOTE_TAKEOVER_S=0: a new main process waits however long the served one is silent; a connection the
//      worker accepted while the served one was idle is closed when the served one sends again (and does not take
//      over at its next idle time); the idle time starts when a slow reply (a relay's) is out, not when it began.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

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
    bool start(const StageHandlers& h, const StageBuffers& buf, const std::string& token = "") {
        std::mt19937 rng(std::random_device{}());
        for (int attempt = 0; attempt < 8 && where.empty(); ++attempt) {
            const int port = 20000 + (int) (rng() % 40000);
            served = -1;
            t = std::thread([&, port, token] { served = serve_stage("127.0.0.1", port, token, h, buf, st, &stop_flag); });
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

// a worker reads it when it starts listening
void set_takeover(const char* s) {
#if defined(_WIN32)
    _putenv_s("STRATA_REMOTE_TAKEOVER_S", s);
#else
    setenv("STRATA_REMOTE_TAKEOVER_S", s, 1);
#endif
}

// a peer that connects to a worker on loopback and sends its hello in parts, when the test says; a thread waits (up
// to 15 s) for the worker to close it.  The sockets are started: a StageClient connected before.
struct Silent {
#if defined(_WIN32)
    using sock = SOCKET;
    static constexpr sock kNone = INVALID_SOCKET;
    static void shut(sock s) { (void) ::shutdown(s, SD_BOTH); }
    static void close_(sock s) { (void) ::closesocket(s); }
#else
    using sock = int;
    static constexpr sock kNone = -1;
    static void shut(sock s) { (void) ::shutdown(s, SHUT_RDWR); }
    static void close_(sock s) { (void) ::close(s); }
#endif
    sock s = kNone;
    std::thread t;
    std::atomic<bool> closed{false};   // the worker closed it (its recv ended before its own 15 s)
    std::chrono::steady_clock::time_point at;
    bool open(const std::string& where) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((unsigned short) std::atoi(where.substr(where.rfind(':') + 1).c_str()));
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == kNone || ::connect(s, (const sockaddr*) &a, sizeof a) != 0) return false;
#if defined(_WIN32)
        const DWORD ms = 15000;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &ms, sizeof ms);
#else
        timeval tv{};
        tv.tv_sec = 15;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
        t = std::thread([this] {
            const auto t0 = std::chrono::steady_clock::now();
            char c;
            (void) ::recv(s, &c, 1, 0);
            at = std::chrono::steady_clock::now();
            closed = at - t0 < std::chrono::seconds(14);
        });
        return true;
    }
    // bytes of a hello, a part at a time (to a connection the worker closed: dropped, without SIGPIPE)
    void send_part(const std::vector<char>& b, size_t from, size_t to) {
#if defined(_WIN32)
        (void) ::send(s, b.data() + from, (int) (to - from), 0);
#else
        (void) ::send(s, b.data() + from, to - from, MSG_NOSIGNAL);
#endif
    }
    void close() {
        if (s == kNone) return;
        shut(s);
        if (t.joinable()) t.join();
        close_(s);
        s = kNone;
    }
    ~Silent() { close(); }
};
}  // namespace

int main() {
    std::printf("stage_link_test\n");
    set_takeover("1");   // (section 5: before the first worker reads it)
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
    std::atomic<int> disconnects{0};
    h.disconnected = [&] { ++disconnects; };
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
    check(!stage_link_lost(err), "... a worker's error is not a lost link");
    c.close();
    wk2.stop();

    // 5. a new main process takes over a connection silent for STRATA_REMOTE_TAKEOVER_S (A's PC lost power: its socket
    //    is never closed)
    worker_vision = 1;
    Worker wk3;
    if (!wk3.start(h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    auto with_token = [](StageHello x, const char* t) {
        std::memset(x.token, 0, sizeof x.token);
        std::memcpy(x.token, t, std::strlen(t));
        return x;
    };
    using SteadyClock = std::chrono::steady_clock;
    // a main process connecting on a thread of its own (it waits while the worker leaves it in the backlog)
    struct Pending {
        StageClient cl;
        std::thread t;
        std::atomic<bool> done{false};
        bool ok = false;
        std::string err;
        SteadyClock::time_point at;
    };
    auto connect_later = [&](Pending& p, StageHello mine, const std::string& at) {
        p.t = std::thread([&p, at, mine] {
            StageHello theirs;
            p.ok = p.cl.connect(at, mine, theirs, p.err);
            p.at = SteadyClock::now();
            p.done = true;
        });
    };
    StageClient a, bad;
    check(greet(a, wk3.where, with_token(hello_of(), "tok")) && a.reset(err), "main A is served");
    const int disc0 = disconnects.load();
    check(!greet(bad, wk3.where, with_token(hello_of(), "nope")) && contains(err, "wrong token"),
          "a main with a wrong token is refused (once A is idle)");
    check(!stage_link_lost(err), "... a refusal is not a lost link");
    StageHello other = with_token(hello_of(), "tok");
    other.pack_hash ^= 1;
    check(!greet(bad, wk3.where, other) && contains(err, "differ"), "... and one with another model");
    err.clear();
    check(a.reset(err) && disconnects.load() == disc0, "... A is still served, never disconnected");
    Pending b;
    connect_later(b, with_token(hello_of(), "tok"), wk3.where);
    bool a_ok = true;
    SteadyClock::time_point a_last = SteadyClock::now();
    for (int i = 0; i < 6; ++i) {   // A busy: a message every 0.5 s, for 3 s (STRATA_REMOTE_TAKEOVER_S is 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        a_ok = a_ok && a.reset(err);
        a_last = SteadyClock::now();
    }
    check(a_ok && !b.done.load() && disconnects.load() == disc0,
          "main B waits while A sends a message every 0.5 s (A served throughout)");
    for (int i = 0; i < 500 && !b.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b.t.join();
    const double a_quiet_ms = std::chrono::duration<double, std::milli>(b.at - a_last).count();
    check(b.ok, "once A is silent past it, B, valid, takes the worker over");
    check(a_quiet_ms >= 1000 && a_quiet_ms < 4000, "... after A's 1 s without a message (not its keepalive or timeouts)");
    check(b.cl.reset(err) && b.cl.run(1, &tok, 0, rows, kHbf, back, kHbf, err) &&
              std::memcmp(rows, back, sizeof rows) == 0,
          "... and B is served");
    for (int i = 0; i < 100 && disconnects.load() == disc0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() == disc0 + 1, "A ended as a closed connection does (the disconnect step ran once)");
    err.clear();
    check(!a.reset(err) && stage_link_lost(err), "A's next message fails as a lost link");
    check(b.cl.reset(err), "... B is still served");
    // a served connection that closes: the one waiting is served at once, not after the idle time
    Pending c3;
    connect_later(c3, with_token(hello_of(), "tok"), wk3.where);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const bool b_last = b.cl.reset(err);
    const auto b_closed = SteadyClock::now();
    b.cl.close();
    for (int i = 0; i < 500 && !c3.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    c3.t.join();
    const double c_ms = std::chrono::duration<double, std::milli>(c3.at - b_closed).count();
    check(b_last && c3.ok && c_ms < 900, "B closes: the main waiting is served at once (under the 1 s idle time)");
    check(c3.cl.reset(err), "... and its messages are answered");
    a.close();
    c3.cl.close();
    bad.close();
    wk3.stop();
    err.clear();
    check(!greet(a, wk3.where, with_token(hello_of(), "tok")) && contains(err, "cannot connect") && stage_link_lost(err),
          "nothing listening: a lost link (worth trying again)");
    a.close();
    err.clear();
    check(!greet(a, "name.invalid:5000", with_token(hello_of(), "tok")) && contains(err, "cannot resolve") &&
              stage_link_lost(err),
          "a name that does not resolve: a lost link (a name service may answer later)");
    a.close();
    check(!greet(a, "no-port", with_token(hello_of(), "tok")) && !stage_link_lost(err),
          "an address without a port: not a lost link");
    a.close();
    auto ms_between = [](SteadyClock::time_point from, SteadyClock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };

    // 6. STRATA_REMOTE_TAKEOVER_S=0: never taken over, however long the served one is silent
    set_takeover("0");
    Worker wk4;
    const bool wk4_up = wk4.start(h, buf, "tok");
    set_takeover("1");
    if (!wk4_up) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet(a, wk4.where, with_token(hello_of(), "tok")) && a.reset(err),
          "STRATA_REMOTE_TAKEOVER_S=0: main A is served");
    int disc = disconnects.load();
    Pending b4;
    connect_later(b4, with_token(hello_of(), "tok"), wk4.where);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    err.clear();
    check(!b4.done.load() && disconnects.load() == disc && a.reset(err),
          "... B waits while A is silent for 2.5 s (past the 1 s above): A still served");
    const auto a4_closed = SteadyClock::now();
    a.close();
    for (int i = 0; i < 500 && !b4.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b4.t.join();
    check(b4.ok && ms_between(a4_closed, b4.at) < 900, "... and is served at once when A closes");
    b4.cl.close();
    wk4.stop();

    // a connection accepted while A was idle (its hello not in yet), then A sends again: that one is closed at once
    // (its main process connects again), and does not take over at A's next idle time
    Worker wk5;
    if (!wk5.start(h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet(a, wk5.where, with_token(hello_of(), "tok")) && a.reset(err), "main A is served");
    disc = disconnects.load();
    // a valid hello (header, then the hello), of which the peer sends 8 bytes first
    std::vector<char> hello_bytes(sizeof(StageHeader) + sizeof(StageHello));
    {
        StageHeader hh;
        hh.type = (uint32_t) StageMsg::Hello;
        hh.bytes = sizeof(StageHello);
        const StageHello hl = with_token(hello_of(), "tok");
        std::memcpy(hello_bytes.data(), &hh, sizeof hh);
        std::memcpy(hello_bytes.data() + sizeof hh, &hl, sizeof hl);
    }
    Silent quiet;
    check(quiet.open(wk5.where), "a peer connects and sends 8 bytes of its hello");
    quiet.send_part(hello_bytes, 0, 8);
    std::this_thread::sleep_for(std::chrono::milliseconds(1800));   // A silent past 1 s: the peer accepted meanwhile
    err.clear();
    const auto a5_sent = SteadyClock::now();
    const bool a5_ok = a.reset(err);
    const double a5_ms = ms_between(a5_sent, SteadyClock::now());
    for (int i = 0; i < 300 && !quiet.closed.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(a5_ok && a5_ms < 1000, "A sends again: it is answered at once (the peer's part of a hello not waited for)");
    check(quiet.closed.load() && ms_between(a5_sent, quiet.at) < 2000,
          "... and the peer accepted meanwhile is closed then (not at its hello's 10 s)");
    quiet.send_part(hello_bytes, 8, hello_bytes.size());   // the rest of its hello, too late
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    err.clear();
    check(a.reset(err) && disconnects.load() == disc,
          "... its hello's rest does not take over at A's next 1 s without a message: A still served");
    quiet.close();
    a.close();
    wk5.stop();

    // the idle time starts when a slow reply is out (a relay's, waiting on its next worker), not when it began
    StageHandlers slow = h;
    slow.prefill_reply = [](float*, int64_t, int64_t, std::string&) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        return true;
    };
    Worker wk6;
    if (!wk6.start(slow, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet(a, wk6.where, with_token(hello_of(), "tok")) && a.reset(err), "a relay's main A is served");
    disc = disconnects.load();
    Pending b6;
    connect_later(b6, with_token(hello_of(), "tok"), wk6.where);
    const int64_t ptok[2] = {1, 2};
    const float pin[2 * kD] = {1, 2, 3, 4, 5, 6, 7, 8};
    float pout[2 * kD] = {};
    err.clear();
    bool a6_ok = a.prefill_send(ptok, 2, 0, 0, pin, kD, 0, err) && a.prefill_recv(pout, kD, 2, 0, err) &&
                 std::memcmp(pin, pout, sizeof pin) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(400));   // A's own part before its next message
    a6_ok = a6_ok && a.reset(err);
    check(a6_ok && !b6.done.load() && disconnects.load() == disc,
          "a chunk's reply out after 1.5 s, A's next message 0.4 s later: A still served, B waits");
    for (int i = 0; i < 500 && !b6.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b6.t.join();
    check(b6.ok, "... B takes over once A is silent past 1 s");
    b6.cl.close();
    a.close();
    wk6.stop();

    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
