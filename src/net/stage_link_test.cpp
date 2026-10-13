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
//      over at its next idle time); the idle time starts when a slow reply (a relay's) is out, not when it began;
//   7. StageClient::alive: true on an idle served link, and it reads nothing (the next reply is in step); false, as
//      a lost link, once the worker's end closed it (a take-over by another main process, the worker stopped) or sent
//      bytes no message asked for, and the link is marked failed; no check blocks;
//   8. connect() with a deadline: a hello left unanswered and a connect no host answers are given up at it, as lost
//      links; a stop flag ends the wait at once (also before a try that would fail at once), not as a lost link; a
//      worker that answers is served as without one;
//   9. StageHandlers::idle (a relay's look at its next worker): not while the served main process sends, about once
//      a second while it is idle; false ends that connection, and the main process's next message finds it lost;
//  10. STRATA_STAGE_ROWS_BF16 (StageHello::row_format): fp32 -> bf16 rounds to nearest even (ties, infinities, NaNs,
//      subnormals, every bf16 value back to itself, random values against a reference); a window and prompt chunks
//      (one spanning several of the link's pieces, two in flight, `skip`) go out and come back as the bf16-rounded
//      rows, half the bytes on the wire; "chunks" leaves windows fp32; fp32 stays byte-for-byte as before (4 bytes a
//      float, the rows exact); a worker refuses a main process whose rows travel otherwise, both ways, saying which
//      side has which; a main process refuses a worker that does not check (one from before row_format) itself; a
//      hello from before row_format reads as fp32; the variable's values.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
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

// STRATA_STAGE_ROWS_BF16 (nullptr: unset)
void set_rows_env(const char* s) {
#if defined(_WIN32)
    _putenv_s("STRATA_STAGE_ROWS_BF16", s != nullptr ? s : "");
#else
    if (s != nullptr) setenv("STRATA_STAGE_ROWS_BF16", s, 1);
    else unsetenv("STRATA_STAGE_ROWS_BF16");
#endif
}

float f32_of(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}
// fp32 -> bf16 the long way (finite values whose rounding stays finite): the nearer of the two bf16 values around
// it, a tie to the one with an even last bit
uint16_t bf16_reference(float f) {
    const uint16_t lo = (uint16_t) (bits_of(f) >> 16), hi = (uint16_t) (lo + 1);
    const double dl = std::fabs((double) stage_f32_from_bf16(lo) - (double) f);
    const double dh = std::fabs((double) stage_f32_from_bf16(hi) - (double) f);
    return dl < dh ? lo : dh < dl ? hi : (lo & 1) != 0 ? hi : lo;
}
// rows as they come back over a bf16 link
std::vector<float> bf16_rounded(const float* p, size_t n) {
    std::vector<float> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = stage_f32_from_bf16(stage_bf16_from_f32(p[i]));
    return r;
}
bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

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

// a worker that answers a hello with hello_of() and then sends `extra`, bytes no message asked for (a link out of
// step); it keeps the connection open until stop().  The sockets are started: a StageClient connected before.
struct Chatty {
#if defined(_WIN32)
    using sock_len = int;
#else
    using sock_len = socklen_t;
#endif
    Silent::sock l = Silent::kNone, s = Silent::kNone;
    std::thread t;
    std::string where;
    static bool recv_n(Silent::sock fd, void* p, size_t n) {
        char* c = (char*) p;
        while (n > 0) {
            const long long r = ::recv(fd, c, (int) n, 0);
            if (r <= 0) return false;
            c += r;
            n -= (size_t) r;
        }
        return true;
    }
    static void send_n(Silent::sock fd, const void* p, size_t n) {
#if defined(_WIN32)
        (void) ::send(fd, (const char*) p, (int) n, 0);
#else
        (void) ::send(fd, p, n, MSG_NOSIGNAL);
#endif
    }
    // a loopback port of its own, listening: `where`
    static Silent::sock listen_any(std::string& where) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = 0;   // any free port
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        Silent::sock l = ::socket(AF_INET, SOCK_STREAM, 0);
        sock_len len = (sock_len) sizeof a;
        if (l == Silent::kNone || ::bind(l, (const sockaddr*) &a, sizeof a) != 0 || ::listen(l, 4) != 0 ||
            ::getsockname(l, (sockaddr*) &a, &len) != 0) {
            if (l != Silent::kNone) Silent::close_(l);
            return Silent::kNone;
        }
        where = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));
        return l;
    }
    bool start(const std::string& extra) {
        if ((l = listen_any(where)) == Silent::kNone) return false;
        t = std::thread([this, extra] {
            s = ::accept(l, nullptr, nullptr);
            if (s == Silent::kNone) return;
            StageHeader h;
            StageHello theirs;
            if (!recv_n(s, &h, sizeof h) || !recv_n(s, &theirs, sizeof theirs)) return;
            StageHeader r;
            r.type = (uint32_t) StageMsg::HelloOk;
            r.bytes = sizeof(StageHello);
            const StageHello mine = hello_of();
            send_n(s, &r, sizeof r);
            send_n(s, &mine, sizeof mine);
            if (!extra.empty()) send_n(s, extra.data(), extra.size());
        });
        return true;
    }
    void stop() {
        if (l != Silent::kNone) {   // first: an accept still waiting (no client came) ends, not the test
            Silent::shut(l);
            Silent::close_(l);
            l = Silent::kNone;
        }
        if (t.joinable()) t.join();
        if (s != Silent::kNone) Silent::close_(s);
        s = Silent::kNone;
    }
    ~Chatty() { stop(); }
};

// a port that takes connections (the system completes them in its backlog) and never answers one: a worker that
// leaves a hello unanswered, as one serving another main process does
struct Mute {
    Silent::sock l = Silent::kNone;
    std::string where;
    bool start() { return (l = Chatty::listen_any(where)) != Silent::kNone; }
    ~Mute() {
        if (l != Silent::kNone) Silent::close_(l);
    }
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

    // 7. alive(): the link between requests, checked with nothing sent and without blocking
    Worker wk7;
    if (!wk7.start(h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    double alive_worst_ms = 0;
    auto alive_timed = [&](StageClient& cl) {
        const auto t0 = SteadyClock::now();
        const bool up = cl.alive(err);
        alive_worst_ms = std::max(alive_worst_ms, ms_between(t0, SteadyClock::now()));
        return up;
    };
    // the first check that finds it down, within 2 s (the worker's end may take a moment to arrive)
    auto alive_until_down = [&](StageClient& cl) {
        for (int i = 0; i < 200; ++i) {
            if (!alive_timed(cl)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    err.clear();
    bool up = greet(a, wk7.where, with_token(hello_of(), "tok")) && a.reset(err);
    for (int i = 0; i < 20 && up; ++i) up = alive_timed(a);
    check(up && err.empty(), "an idle served link is alive (20 checks)");
    check(a.reset(err) && a.alive(err) && a.reset(err),
          "... the check sends nothing: the next messages and their replies are in step");
    Pending b7;
    connect_later(b7, with_token(hello_of(), "tok"), wk7.where);
    for (int i = 0; i < 500 && !b7.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b7.t.join();
    check(b7.ok, "main B takes the worker over once A is silent past 1 s");
    check(alive_until_down(a) && stage_link_lost(err),
          "A's check finds its connection ended, as a lost link (nothing sent)");
    check(!a.connected() && !a.alive(err) && stage_link_lost(err) && !a.reset(err) && stage_link_lost(err),
          "... and marks the link failed: later calls fail at once, as lost");
    check(b7.cl.alive(err) && b7.cl.reset(err), "B, served, is alive");
    wk7.stop();
    check(alive_until_down(b7.cl) && stage_link_lost(err), "the worker stopped: B's check finds the link lost");
    b7.cl.close();
    a.close();
    Chatty idle_peer, chatty;
    err.clear();
    up = idle_peer.start("") && greet(a, idle_peer.where, hello_of());
    for (int i = 0; i < 20 && up; ++i) {
        up = alive_timed(a);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    check(up, "a peer that sends nothing after its hello: alive (20 checks over 0.2 s)");
    a.close();
    idle_peer.stop();
    err.clear();
    check(chatty.start(std::string("\x01\x02\x03", 3)) && greet(a, chatty.where, hello_of()) &&
              alive_until_down(a) && contains(err, "out of step") && stage_link_lost(err) && !a.connected(),
          "a peer that sends bytes no message asked for: not alive, as lost (out of step)");
    a.close();
    chatty.stop();
    check(alive_worst_ms < 100, "no check above blocked (each under 100 ms)");

    // 8. connect() with a deadline (timeout_s) and a stop flag: the connect and the hello's reply together
    Mute mute;
    if (!mute.start()) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    auto timed_connect = [&](StageClient& cl, const std::string& at, StageHello mine, int timeout_s,
                             const std::atomic<bool>* stop, double& ms) {
        cl.close();
        err.clear();
        const auto t0 = SteadyClock::now();
        const bool ok = cl.connect(at, mine, peer, err, timeout_s, stop);
        ms = ms_between(t0, SteadyClock::now());
        return ok;
    };
    double ms = 0;
    check(!timed_connect(a, mute.where, hello_of(), 1, nullptr, ms) && ms >= 900 && ms < 3000 &&
              contains(err, "no reply to the hello") && stage_link_lost(err),
          "a hello left unanswered: given up after the 1 s, as a lost link (worth trying again)");
    std::atomic<bool> stop_now{false};
    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        stop_now = true;
    });
    const bool stopped_ok = timed_connect(a, mute.where, hello_of(), 30, &stop_now, ms);
    stopper.join();
    check(!stopped_ok && ms >= 250 && ms < 1500 && contains(err, "cancelled") && !stage_link_lost(err),
          "... a stop flag set 0.3 s in ends a 30 s wait at once, and is not a lost link");
    check(!timed_connect(a, mute.where, hello_of(), 0, &stop_now, ms) && ms < 1000 && contains(err, "cancelled"),
          "... one set before the call: no wait (a stop flag without a deadline)");
    // a try that fails at once (a name that does not resolve: .invalid, RFC 6761) never reaches a poll that looks at
    // the flag: a caller that tries again every few seconds must still see its STOP
    check(!timed_connect(a, "stage-link-test.invalid:9", hello_of(), 5, &stop_now, ms) && ms < 1000 &&
              stage_wait_cancelled(err) && !stage_link_lost(err),
          "... and before a try that would fail at once (a name that does not resolve): cancelled");
    // SYNs no host answers (TEST-NET-1): given up at the deadline, not after the system's tries (20 s to 2 min); a
    // network that refuses it at once ends sooner, also as a lost link (the line after the check says which it was)
    check(!timed_connect(a, "192.0.2.1:9", hello_of(), 1, nullptr, ms) && ms < 3000 && stage_link_lost(err),
          "a connect no host answers: given up within the 1 s, as a lost link");
    std::printf("    (%s)\n", contains(err, "timed out") ? "it timed out: the deadline ended it"
                                                         : "the network refused it at once: the deadline not reached");
    Worker wk8;
    if (!wk8.start(h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    std::atomic<bool> never{false};
    check(timed_connect(a, wk8.where, with_token(hello_of(), "tok"), 5, &never, ms) && a.reset(err) &&
              a.run(1, &tok, 0, rows, kHbf, back, kHbf, err) && std::memcmp(rows, back, sizeof rows) == 0,
          "a worker that answers: served within the deadline, the link blocking again for its messages");
    a.close();
    wk8.stop();

    // 9. h.idle (a relay's look at its next worker): only while the served main process is idle, about once a
    //    second; false ends that connection, which the main process's next message finds lost
    std::atomic<int> idle_looks{0};
    std::atomic<bool> next_gone{false};
    StageHandlers relay_h = h;
    relay_h.idle = [&](std::string& e) {
        ++idle_looks;
        if (!next_gone.load()) return true;
        e = "the next worker went away (test)";
        return false;
    };
    Worker wk9;
    if (!wk9.start(relay_h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet(a, wk9.where, with_token(hello_of(), "tok")) && a.reset(err), "a relay's main A is served");
    bool busy_ok = true;
    for (int i = 0; i < 6; ++i) {   // a message every 0.3 s, for 1.8 s
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        busy_ok = busy_ok && a.reset(err);
    }
    check(busy_ok && idle_looks.load() == 0, "no look while A sends a message every 0.3 s");
    std::this_thread::sleep_for(std::chrono::milliseconds(2600));
    const int looks = idle_looks.load();
    check(looks >= 1 && looks <= 3 && a.reset(err), "A idle for 2.6 s: about one look a second, and A still served");
    disc = disconnects.load();
    next_gone = true;
    const auto gone_at = SteadyClock::now();
    for (int i = 0; i < 300 && disconnects.load() == disc; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() == disc + 1 && ms_between(gone_at, SteadyClock::now()) < 2500,
          "the next worker gone while A is idle: A's connection ends within the next look");
    err.clear();
    check(!a.reset(err) && stage_link_lost(err), "... A's next message finds it lost (its main process connects again)");
    next_gone = false;
    check(greet(a, wk9.where, with_token(hello_of(), "tok")) && a.reset(err), "... and connected again, A is served");
    a.close();
    wk9.stop();

    // 10. STRATA_STAGE_ROWS_BF16: the rows as bf16 on the wire, fp32 at both ends
    {
        check(stage_bf16_from_f32(1.0f) == 0x3f80 && stage_f32_from_bf16(0x3f80) == 1.0f &&
                  stage_bf16_from_f32(-0.0f) == 0x8000 && stage_bf16_from_f32(0.0f) == 0,
              "bf16: 1, 0 and -0 are exact");
        check(stage_bf16_from_f32(f32_of(0x3f808000u)) == 0x3f80 && stage_bf16_from_f32(f32_of(0x3f818000u)) == 0x3f82 &&
                  stage_bf16_from_f32(f32_of(0xbf818000u)) == 0xbf82,
              "bf16: a tie rounds to the even neighbour (down, up, and by magnitude when negative)");
        check(stage_bf16_from_f32(f32_of(0x3f808001u)) == 0x3f81 && stage_bf16_from_f32(f32_of(0x3f807fffu)) == 0x3f80,
              "bf16: off a tie, to the nearer one");
        check(stage_bf16_from_f32(f32_of(0x00008000u)) == 0x0000 && stage_bf16_from_f32(f32_of(0x00018000u)) == 0x0002 &&
                  stage_bf16_from_f32(f32_of(0x00018001u)) == 0x0002,
              "bf16: subnormals round the same way");
        check(stage_bf16_from_f32(INFINITY) == 0x7f80 && stage_bf16_from_f32(-INFINITY) == 0xff80 &&
                  stage_bf16_from_f32(f32_of(0x7f7fffffu)) == 0x7f80 && stage_bf16_from_f32(f32_of(0x7f7f7fffu)) == 0x7f7f,
              "bf16: infinities stay, the largest floats round to one, the largest bf16 stays");
        check(std::isnan(stage_f32_from_bf16(stage_bf16_from_f32(f32_of(0x7f800001u)))) &&
                  std::isnan(stage_f32_from_bf16(stage_bf16_from_f32(f32_of(0xffc00000u)))) &&
                  std::isnan(stage_f32_from_bf16(stage_bf16_from_f32(f32_of(0x7fffffffu)))),
              "bf16: a NaN stays a NaN (also one whose payload is in the low bits only)");
        bool every = true;
        for (uint32_t b = 0; b < 65536; ++b) {
            const float f = stage_f32_from_bf16((uint16_t) b);
            every = every && (std::isnan(f) ? std::isnan(stage_f32_from_bf16(stage_bf16_from_f32(f)))
                                            : stage_bf16_from_f32(f) == (uint16_t) b);
        }
        check(every, "bf16: every bf16 value comes back as itself (NaNs as NaNs)");
        std::mt19937 rng(1748);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_int_distribution<int> scale(-30, 30);
        bool like_ref = true, within = true;
        for (int i = 0; i < 200000; ++i) {
            const float f = std::ldexp(normal(rng), scale(rng));
            const uint16_t b = stage_bf16_from_f32(f);
            like_ref = like_ref && b == bf16_reference(f);
            within = within && std::fabs(stage_f32_from_bf16(b) - f) <= std::fabs(f) * (1.0f / 256.0f);
        }
        check(like_ref, "bf16: 200,000 random values round as the reference (nearest, ties to even)");
        check(within, "... each within 2^-8 of its value (half a unit in bf16's 8 significant bits)");
    }
    {
        // a worker whose hello says how its rows travel (worker_rows), with a prompt buffer for a chunk of three of the
        // link's pieces (64K floats each)
        constexpr int64_t kBig = 40000;   // rows of kD floats: 160,000 floats a chunk
        std::vector<float> big_in0((size_t) (kBig * kD)), big_in1(big_in0.size()), big_out0(big_in0.size()),
            big_out1(big_in0.size());
        StageBuffers bbuf = buf;
        bbuf.pf_in[0] = big_in0.data();
        bbuf.pf_in[1] = big_in1.data();
        bbuf.pf_out[0] = big_out0.data();
        bbuf.pf_out[1] = big_out1.data();
        bbuf.pf_floats = big_in0.size();
        std::atomic<uint32_t> worker_rows{kStageRowsBf16Chunks | kStageRowsBf16Windows};
        StageHandlers hb = h;
        hb.hello = [&](const StageHello&, StageHello& mine, std::string&) {
            mine = hello_of();
            mine.row_format = worker_rows.load();
            return true;
        };
        Worker wk10;
        if (!wk10.start(hb, bbuf)) {
            std::printf("stage_link_test: no loopback port to listen on\n");
            return 1;
        }
        auto rows_hello = [](uint32_t f) {
            StageHello x = hello_of();
            x.row_format = f;
            return x;
        };
        const uint32_t kBoth = kStageRowsBf16Chunks | kStageRowsBf16Windows;
        // a window of kMaxT rows: values bf16 cannot hold (a tie, a subnormal, large and small), and exact ones
        float win[kMaxT * kHbf], wback[kMaxT * kHbf];
        const float wvals[kMaxT * kHbf] = {3.14159265f, -2.71828183f, 1.0f, -0.0f,
                                           f32_of(0x3f818000u), f32_of(0x3f808001u), 1e-40f, -1e-3f,
                                           123456.789f, -6.5e-5f, 3.0e38f, 0.1f,
                                           0.33333334f, -1.5f, 7.0e-30f, 65504.0f};
        std::memcpy(win, wvals, sizeof win);
        const int32_t wtok[kMaxT] = {1, 2, 3, 4};
        // two prompt chunks of kBig rows, normally distributed at mixed scales
        std::vector<float> ch_a((size_t) (kBig * kD)), ch_b(ch_a.size()), back_a(ch_a.size(), -9.0f),
            back_b(ch_a.size(), -9.0f);
        std::mt19937 rng(10);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (size_t i = 0; i < ch_a.size(); ++i) {
            ch_a[i] = std::ldexp(normal(rng), (int) (i % 13) - 6);
            ch_b[i] = normal(rng) * 37.0f;
        }
        std::vector<int64_t> ctok((size_t) kBig);
        for (int64_t i = 0; i < kBig; ++i) ctok[(size_t) i] = i;
        const int64_t skip = 3;
        const size_t back_floats = (size_t) ((kBig - skip) * kD);

        StageClient b;
        check(greet(b, wk10.where, rows_hello(kBoth)) && peer.row_format == kBoth && b.row_format() == kBoth,
              "STRATA_STAGE_ROWS_BF16=1 on both: served, both hellos say bf16 rows");
        uint64_t out0 = b.stats().bytes_out, in0 = b.stats().bytes_in;
        err.clear();
        bool ok = b.run((int) kMaxT, wtok, 0, win, (size_t) (kMaxT * kHbf), wback, (size_t) (kMaxT * kHbf), err);
        const std::vector<float> wround = bf16_rounded(win, (size_t) (kMaxT * kHbf));
        check(ok && same_bits(wback, wround.data(), wround.size()) && !same_bits(wback, win, wround.size()),
              "a window comes back as its bf16-rounded rows (fp32 at both ends)");
        check(b.stats().bytes_out - out0 == sizeof(StageHeader) + kMaxT * 4 + kMaxT * kHbf * 2 &&
                  b.stats().bytes_in - in0 == sizeof(StageHeader) + kMaxT * kHbf * 2,
              "... 2 bytes a float on the wire, each way");
        out0 = b.stats().bytes_out;
        in0 = b.stats().bytes_in;
        ok = b.prefill_send(ctok.data(), kBig, 0, 0, ch_a.data(), kD, skip, err) &&
             b.prefill_send(ctok.data(), kBig, kBig, 0, ch_b.data(), kD, 0, err) &&
             b.prefill_recv(back_a.data(), kD, kBig, skip, err) && b.prefill_recv(back_b.data(), kD, kBig, 0, err);
        const std::vector<float> ra = bf16_rounded(ch_a.data(), ch_a.size()), rb = bf16_rounded(ch_b.data(), ch_b.size());
        bool untouched = true;
        for (size_t i = 0; i < (size_t) (skip * kD); ++i) untouched = untouched && back_a[i] == -9.0f;
        check(ok && same_bits(back_a.data() + skip * kD, ra.data() + skip * kD, back_floats) && untouched &&
                  same_bits(back_b.data(), rb.data(), rb.size()),
              "two prompt chunks in flight (three pieces each): their bf16-rounded rows, from `skip` on");
        check(b.stats().bytes_out - out0 == 2 * (sizeof(StageHeader) + kBig * 8 + kBig * kD * 2) &&
                  b.stats().bytes_in - in0 == 2 * sizeof(StageHeader) + (back_floats + kBig * kD) * 2,
              "... 2 bytes a float on the wire, each way");
        err.clear();
        check(b.reset(err) && b.run(1, wtok, 0, win, kHbf, wback, kHbf, err) && same_bits(wback, wround.data(), kHbf),
              "... and the link is in step after them");
        b.close();

        worker_rows = kStageRowsBf16Chunks;
        check(greet(b, wk10.where, rows_hello(kStageRowsBf16Chunks)) && b.row_format() == kStageRowsBf16Chunks,
              "STRATA_STAGE_ROWS_BF16=chunks on both: served");
        out0 = b.stats().bytes_out;
        err.clear();
        std::fill(back_a.begin(), back_a.end(), -9.0f);
        ok = b.run((int) kMaxT, wtok, 0, win, (size_t) (kMaxT * kHbf), wback, (size_t) (kMaxT * kHbf), err);
        check(ok && same_bits(wback, win, (size_t) (kMaxT * kHbf)) &&
                  b.stats().bytes_out - out0 == sizeof(StageHeader) + kMaxT * 4 + kMaxT * kHbf * 4,
              "... a window travels as fp32 (exact, 4 bytes a float)");
        ok = b.prefill_send(ctok.data(), kBig, 0, 0, ch_a.data(), kD, 0, err) &&
             b.prefill_recv(back_a.data(), kD, kBig, 0, err);
        check(ok && same_bits(back_a.data(), ra.data(), ra.size()), "... a prompt chunk as bf16");
        b.close();

        worker_rows = 0;
        check(greet(b, wk10.where, rows_hello(0)) && b.row_format() == 0, "unset on both: served, fp32 rows");
        out0 = b.stats().bytes_out;
        in0 = b.stats().bytes_in;
        err.clear();
        std::fill(back_a.begin(), back_a.end(), -9.0f);
        ok = b.run((int) kMaxT, wtok, 0, win, (size_t) (kMaxT * kHbf), wback, (size_t) (kMaxT * kHbf), err) &&
             same_bits(wback, win, (size_t) (kMaxT * kHbf)) &&
             b.prefill_send(ctok.data(), kBig, 0, 0, ch_a.data(), kD, skip, err) &&
             b.prefill_recv(back_a.data(), kD, kBig, skip, err) &&
             same_bits(back_a.data() + skip * kD, ch_a.data() + skip * kD, back_floats);
        check(ok && b.stats().bytes_out - out0 ==
                        sizeof(StageHeader) * 2 + kMaxT * 4 + kMaxT * kHbf * 4 + kBig * 8 + kBig * kD * 4 &&
                  b.stats().bytes_in - in0 == sizeof(StageHeader) * 2 + kMaxT * kHbf * 4 + back_floats * 4,
              "... the rows exact, 4 bytes a float, as before row_format");
        b.close();

        worker_rows = kBoth;
        check(!greet(b, wk10.where, rows_hello(0)) && contains(err, "the worker refused:") && contains(err, "differ") &&
                  contains(err, "main fp32, worker bf16") && contains(err, "STRATA_STAGE_ROWS_BF16") &&
                  !stage_link_lost(err),
              "a main with fp32 rows is refused by a worker with bf16, saying which has which (not a lost link)");
        std::printf("    (%s)\n", err.c_str());
        worker_rows = 0;
        check(!greet(b, wk10.where, rows_hello(kBoth)) && contains(err, "differ") &&
                  contains(err, "main bf16, worker fp32"),
              "... and one with bf16 by a worker with fp32");
        worker_rows = kStageRowsBf16Chunks;
        check(!greet(b, wk10.where, rows_hello(kBoth)) && contains(err, "differ") &&
                  contains(err, "main bf16, worker bf16 prompt chunks, fp32 windows"),
              "... and one with every row bf16 by a worker with bf16 chunks only");
        b.close();
        wk10.stop();
        // a worker from before row_format answers any hello (hello_of(): fp32): the main process refuses it itself
        Chatty older;
        check(older.start("") && !greet(b, older.where, rows_hello(kBoth)) && !contains(err, "the worker refused") &&
                  contains(err, "differ") && contains(err, "main bf16, worker fp32") && !b.connected(),
              "a main with bf16 refuses a worker that does not check (fp32 rows) itself");
        b.close();
        older.stop();
        // protocol 5's hello before row_format: a 56-byte build id ("strata remote-stage worker" and the like) whose
        // last 4 bytes, now row_format, it left zero
        StageHello before;
        {
            char* field = reinterpret_cast<char*>(&before) + offsetof(StageHello, build);   // (build and row_format)
            const char* id = "strata remote-stage worker (sycl)";   // the longest the engine wrote: 33 characters
            std::memset(field, 0, 56);
            std::memcpy(field, id, std::strlen(id));
        }
        check(before.row_format == 0 && offsetof(StageHello, row_format) == offsetof(StageHello, build) + 52,
              "a hello from before row_format (its build id in 56 bytes) reads as fp32 rows");
        // the variable
        set_rows_env(nullptr);
        const uint32_t unset = stage_row_format_env();
        set_rows_env("0");
        const uint32_t zero = stage_row_format_env();
        set_rows_env("1");
        const uint32_t one = stage_row_format_env();
        set_rows_env("chunks");
        const uint32_t chunks = stage_row_format_env();
        set_rows_env("yes");
        const uint32_t other = stage_row_format_env();
        set_rows_env(nullptr);
        check(unset == 0 && zero == 0 && one == kBoth && chunks == kStageRowsBf16Chunks && other == 0,
              "STRATA_STAGE_ROWS_BF16: unset and 0 fp32, 1 every row bf16, chunks the chunks, else fp32 (warned)");
        check(stage_row_format_name(0) == "fp32" && stage_row_format_name(kBoth) == "bf16" &&
                  stage_row_format_name(8) == "unknown (8)",
              "... and their names in the logs");
    }

    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
