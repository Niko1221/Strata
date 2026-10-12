// src/net/stage_link_test.cpp - the remote stage's link over loopback, no GPU: a worker (serve_stage) with handlers
// that copy rows, keep a StageCkptStore and record the image positions they get, and a StageClient driving it.  What
// it checks (docs/remote-stage/CHECKPOINTS.md; docs/remote-stage/ENGINEERING.md, "Images"):
//
//   0. the worker's table from what a Positions message carries (stage_positions_fill): the identity, the cells
//      sent, the cells past them;
//   1. the hello: an older side (protocol 4) is refused for its protocol, a main process that would have the worker
//      keep more than kStageCkptMax checkpoints is refused, a main process with --vision refuses a worker without it,
//      a matching one is served;
//   2. Reset / CkptSave / CkptRestore carry their id, position and keep[] whole (int64 ids beyond 32 bits);
//   3. the worker's position: a checkpoint is saved only where the session is (0 after a reset, a chunk's end, a
//      window's first position plus its accepted count, L after a restore); a failed Commit comes back on the
//      CkptSave, and a CkptRestore drops it (the session is replaced);
//   4. CkptSaveOk a=0 (not stored) leaves the store as it was; a restore of an id the store does not hold is the
//      missing-id error (the main process's fallback), any other restore error is not;
//   5. a keep[] longer than kStageCkptMax is a malformed message; a disconnect clears the store;
//   6. Positions carries the table whole (cells x (t, h, w), values past 16 bits) and the identity as no table; a
//      failed Positions comes back on the next reply - also when that is a Reset or a CkptRestore, which drop a failed
//      Commit's error but not the next request's positions - and the link stays usable; a later table replaces a
//      failed one; a handler's success is its return value; a malformed table is the next reply's error, not a reply
//      of its own; a Positions does not move the session's position;
//   7. a worker without a positions handler (started without --vision) answers a Positions with an error naming
//      --vision;
//   8. a new main process waits while the served one sends messages, and takes it over once it has been silent (not
//      closed) for STRATA_REMOTE_TAKEOVER_S (1 s here): B served, A's next message fails as a lost link; one with a
//      wrong token, another model or more checkpoints than a worker keeps is refused and A stays; when the served one
//      closes, the one waiting is served at once; which errors stage_link_lost counts as a lost link;
//   9. STRATA_REMOTE_TAKEOVER_S=0: a new main process waits however long the served one is silent; a connection the
//      worker accepted while the served one was idle is closed when the served one sends again (and does not take
//      over at its next idle time); the idle time starts when a slow reply (a relay's) is out, not when it began;
//  10. StageClient::alive, the main process's check at a request's start (and a relay's of its next worker): true on
//      an idle served link, and it reads nothing (the next reply is in step); false, as a lost link, once the
//      worker's end closed it (a take-over by another main process, the worker stopped) or sent bytes no message
//      asked for, and the link is marked failed; no check blocks;
//  11. connect() with a deadline: a hello left unanswered and a connect no host answers are given up at it, as lost
//      links; a stop flag ends the wait at once (also before a try that would fail at once), not as a lost link; a
//      worker that answers is served as without one;
//  12. StageHandlers::idle (a relay's look at its next worker): not while the served main process sends, about once
//      a second while it is idle; false ends that connection, and the main process's next message finds it lost;
//  13. a worker reads nothing but a hello first (another first message is answered and the connection closed, its
//      body neither waited for nor drained; a hello sent a byte a second is given up at 10 s in all), and nothing
//      after a hello - the first, or a second on a served connection - before its token is taken (a refused one's next
//      headers, one with a body that never comes, neither end the worker nor hold it); a window's or a chunk's T is
//      bounded before any product with it: T = 2^62, whose sizes wrap to 0, ends no worker - before the hello or
//      after one;
//  14. a hello's reply is read whole within connect()'s deadline (a worker that sends part of it is given up at it,
//      or at a stop flag), and an error reply of 2^62 bytes is not read: at the hello an unexpected reply, later a
//      link out of step.
// The handlers run on the worker's serving thread: what they share with the test is behind a mutex or atomic.
#include "strata/net/stage_ckpt_store.hpp"
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
    StageCkptStore store;
    std::vector<int64_t> last_keep;
    int64_t last_id = -1, last_L = -1;
    int pos_calls = 0;   // the positions handler's
    bool pos_identity = false;
    std::vector<int32_t> pos_table;
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
    bool positions_were(int n, bool identity, const std::vector<int32_t>& table) {
        std::lock_guard<std::mutex> lk(mu);
        return pos_calls == n && pos_identity == identity && pos_table == table;
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
    // answer = false: `extra` is the whole reply to the hello (a crafted one)
    bool start(const std::string& extra, bool answer = true) {
        if ((l = listen_any(where)) == Silent::kNone) return false;
        t = std::thread([this, extra, answer] {
            s = ::accept(l, nullptr, nullptr);
            if (s == Silent::kNone) return;
            StageHeader h;
            StageHello theirs;
            if (!recv_n(s, &h, sizeof h) || !recv_n(s, &theirs, sizeof theirs)) return;
            if (answer) {
                StageHeader r;
                r.type = (uint32_t) StageMsg::HelloOk;
                r.bytes = sizeof(StageHello);
                const StageHello mine = hello_of();
                send_n(s, &r, sizeof r);
                send_n(s, &mine, sizeof mine);
            }
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

// a main process's end spoken by hand, to a worker on loopback: headers a StageClient never sends.  Its reads wait
// 5 s at most.
struct Raw {
    Silent::sock s = Silent::kNone;
    bool open(const std::string& where) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((unsigned short) std::atoi(where.substr(where.rfind(':') + 1).c_str()));
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == Silent::kNone || ::connect(s, (const sockaddr*) &a, sizeof a) != 0) return false;
#if defined(_WIN32)
        const DWORD ms = 5000;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*) &ms, sizeof ms);
#else
        timeval tv{};
        tv.tv_sec = 5;
        (void) setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
        return true;
    }
    void send(const void* p, size_t n) { Chatty::send_n(s, p, n); }
    void header(StageMsg type, int64_t a, uint64_t bytes) {
        StageHeader h;
        h.type = (uint32_t) type;
        h.a = a;
        h.bytes = bytes;
        send(&h, sizeof h);
    }
    // the next message, its payload as bytes (up to 1 MiB); false: none within 5 s, or the worker closed it
    bool reply(StageHeader& r, std::string& payload) {
        if (!Chatty::recv_n(s, &r, sizeof r)) return false;
        payload.assign((size_t) std::min<uint64_t>(r.bytes, (uint64_t) 1 << 20), '\0');
        return Chatty::recv_n(s, payload.data(), payload.size());
    }
    // the worker closed it within 4.5 s, what it still sent read and dropped (not a time-out, nor bytes that keep
    // coming).  A reset counts: a worker that closes with bytes it did not read (headers sent behind a refused hello)
    // resets the connection.
    bool closed_by_worker() {
        const auto t0 = std::chrono::steady_clock::now();
        char c[256];
        for (;;) {
            const long long r = ::recv(s, c, (int) sizeof c, 0);
            const bool in_time = std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(4500);
            if (r > 0 && in_time) continue;
            return r <= 0 && in_time;
        }
    }
    ~Raw() {
        if (s != Silent::kNone) Silent::close_(s);
    }
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
    set_takeover("1");   // (section 8: before the first worker reads it)
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
    std::atomic<bool> refuse_save{false}, fail_commit{false}, fail_positions{false}, stale_text{false};
    std::atomic<int> disconnects{0};
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
    h.reset = [&](const std::vector<int64_t>& keep, std::string&) {
        std::lock_guard<std::mutex> lk(w.mu);
        w.last_keep = keep;
        w.store.prune(keep);
        return true;
    };
    h.positions = [&](const int32_t* table, int64_t cells, std::string& e) {
        std::lock_guard<std::mutex> lk(w.mu);
        ++w.pos_calls;
        w.pos_identity = table == nullptr;
        w.pos_table.clear();
        if (table != nullptr) w.pos_table.assign(table, table + cells * 3);
        if (stale_text.load()) e = "a message left from a retry (test)";   // and it succeeds
        // fail_positions, or a table starting at t = -7 (no reply tells when the worker has read a Positions, so a
        // flag flipped right after one is a race; the table's own content is not)
        if (!fail_positions.load() && (table == nullptr || table[0] != -7)) return true;
        e = "the positions failed (test)";
        return false;
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
    Worker wk;
    if (!wk.start(h, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    StageClient c;
    std::string err;
    StageHello peer;
    auto greet_at = [&](const std::string& at, StageHello mine) {
        err.clear();
        c.close();
        return c.connect(at, mine, peer, err);
    };
    auto greet = [&](StageHello mine) { return greet_at(wk.where, mine); };

    // 1. the hello
    StageHello old_side = hello_of();
    old_side.protocol = 4;
    check(!greet(old_side) && contains(err, "the worker refused:") && contains(err, "protocol 4/6"),
          "a protocol 4 main process is refused for its protocol");
    StageHello greedy = hello_of();
    greedy.ckpt_max = kStageCkptMax + 1;
    check(!greet(greedy) && contains(err, "at most 64"),
          "a main process asking for more than 64 checkpoints is refused");
    worker_vision = 0;
    check(!greet(hello_of(1)) && contains(err, "--vision"), "a main with --vision refuses a worker without it");
    check(greet(hello_of(0)), "... one without --vision is served by it");
    worker_vision = 1;
    StageHello mine = hello_of();
    mine.ckpt_max = kStageCkptMax;
    check(greet(mine) && peer.vision == 1, "a matching main process is served");

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

    // 6. image positions: the table whole, the identity as none, a failure outlives a reset and a restore
    const int64_t cells = 40;
    std::vector<int32_t> table((size_t) cells * 3);
    for (int64_t i = 0; i < cells; ++i) {   // a grid's (t, h, w), positions past 16 bits
        table[(size_t) i * 3] = 70000;
        table[(size_t) i * 3 + 1] = 70000 + (int32_t) (i / 8);
        table[(size_t) i * 3 + 2] = 70000 + (int32_t) (i % 8);
    }
    err.clear();
    check(c.positions(table.data(), cells, err) && c.reset({}, err) && w.positions_were(1, false, table),
          "a table arrives whole, before the reset");
    check(c.positions(nullptr, 0, err) && c.reset({}, err) && w.positions_were(2, true, {}),
          "the identity arrives as no table");
    const int32_t one[3] = {5, 5, 5};
    fail_positions = true;
    err.clear();
    bool sent = c.positions(one, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "the positions failed (test)"),
          "a failed Positions comes back on the reset");
    fail_positions = false;
    err.clear();
    check(c.reset({}, err), "... once: the link is usable after it");
    fail_commit = true;
    check(c.commit(1, err) && c.positions(one, 1, err) && c.reset({}, err) && w.positions_were(4, false, {5, 5, 5}),
          "a failed Commit is dropped by the reset (the positions after it applied)");
    const int32_t failing[3] = {-7, -7, -7};
    err.clear();
    sent = c.commit(1, err) && c.positions(failing, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "the positions failed (test)") && !contains(err, "a commit failed"),
          "a failed Commit, then a failed Positions: the reset reports the positions only");
    fail_commit = false;
    err.clear();
    sent = c.positions(failing, 1, err);
    check(sent && c.positions(nullptr, 0, err) && c.reset({}, err) && w.positions_were(7, true, {}),
          "a later table replaces a failed one (a refused request: no reset)");
    stale_text = true;
    check(c.positions(one, 1, err) && c.reset({}, err), "a handler that succeeds with text left in its error: a success");
    stale_text = false;
    std::vector<int32_t> huge((size_t) (((int64_t) 1 << 22) + 1) * 3, 1);
    err.clear();
    sent = c.positions(huge.data(), (int64_t) (huge.size() / 3), err);
    check(sent && !c.reset({}, err) && contains(err, "bad position table size") && w.positions_were(8, false, {5, 5, 5}),
          "a malformed table (over 4M cells): the next reply's error, no handler call");
    check(c.reset({}, err), "... and only that reply's: the link is in step after it");
    check(chunk() && c.positions(one, 1, err) && c.ckpt_save(1, kChunk, {1}, stored, err) && stored,
          "a Positions does not move the session: a checkpoint at the chunk's end after one");
    fail_commit = true;
    err.clear();
    sent = c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
           c.commit(1, err) && c.positions(failing, 1, err);
    check(sent && !c.ckpt_restore(1, kChunk, {1}, missing, err) && !missing &&
              contains(err, "the positions failed (test)") && !contains(err, "a commit failed") && w.has(1),
          "a failed Commit, then a failed Positions: a restore reports the positions only");
    fail_commit = false;
    fail_positions = true;
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.ckpt_restore(1, kChunk, {1}, missing, err) && !missing &&
              contains(err, "the positions failed (test)") && w.has(1),
          "a failed Positions comes back on a restore (not the missing-id error)");
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.ckpt_save(2, kChunk, {1, 2}, stored, err) && contains(err, "the positions failed (test)") &&
              !w.has(2),
          "... and on a CkptSave");
    fail_positions = false;
    check(c.ckpt_restore(1, kChunk, {1}, missing, err), "a restore after them");
    fail_positions = true;
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err) &&
              contains(err, "the positions failed (test)"),
          "... and on the next window");
    fail_positions = false;
    check(c.run(1, wtok.data(), kChunk, win.data(), (size_t) kHbf, wout.data(), (size_t) kHbf, err),
          "... and the next window runs");
    c.close();
    wk.stop();

    // 7. a worker started without --vision: no positions handler
    worker_vision = 0;
    StageHandlers plain = h;
    plain.positions = nullptr;
    Worker wk2;
    if (!wk2.start(plain, buf)) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet_at(wk2.where, hello_of(0)), "a main without --vision is served by a worker without it");
    err.clear();
    sent = c.positions(one, 1, err);
    check(sent && !c.reset({}, err) && contains(err, "--vision"), "a Positions to it is an error naming --vision");
    check(!stage_link_lost(err), "... a worker's error is not a lost link");
    c.close();
    wk2.stop();

    // 8. a new main process takes over a connection silent for STRATA_REMOTE_TAKEOVER_S (A's PC lost power: its socket
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
    // main processes of their own (greet connects c)
    auto greet_with = [&](StageClient& cl, const std::string& at, StageHello mine) {
        err.clear();
        cl.close();
        return cl.connect(at, mine, peer, err);
    };
    float rows1[kHbf] = {1, 2, 3, 4}, back1[kHbf] = {};
    const int32_t tok1 = 7;
    StageClient a, bad;
    check(greet_with(a, wk3.where, with_token(hello_of(), "tok")) && a.reset({}, err), "main A is served");
    const int disc0 = disconnects.load();
    check(!greet_with(bad, wk3.where, with_token(hello_of(), "nope")) && contains(err, "wrong token"),
          "a main with a wrong token is refused (once A is idle)");
    check(!stage_link_lost(err), "... a refusal is not a lost link");
    StageHello other = with_token(hello_of(), "tok");
    other.pack_hash ^= 1;
    check(!greet_with(bad, wk3.where, other) && contains(err, "differ"), "... and one with another model");
    StageHello greedy_b = with_token(hello_of(), "tok");
    greedy_b.ckpt_max = kStageCkptMax + 1;
    check(!greet_with(bad, wk3.where, greedy_b) && contains(err, "conversation checkpoints") &&
              disconnects.load() == disc0,
          "... and one that would keep more checkpoints than a worker may (before it takes over)");
    err.clear();
    check(a.reset({}, err) && disconnects.load() == disc0, "... A is still served, never disconnected");
    Pending b;
    connect_later(b, with_token(hello_of(), "tok"), wk3.where);
    bool a_ok = true;
    SteadyClock::time_point a_last = SteadyClock::now();
    for (int i = 0; i < 6; ++i) {   // A busy: a message every 0.5 s, for 3 s (STRATA_REMOTE_TAKEOVER_S is 1)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        a_ok = a_ok && a.reset({}, err);
        a_last = SteadyClock::now();
    }
    check(a_ok && !b.done.load() && disconnects.load() == disc0,
          "main B waits while A sends a message every 0.5 s (A served throughout)");
    for (int i = 0; i < 500 && !b.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b.t.join();
    const double a_quiet_ms = std::chrono::duration<double, std::milli>(b.at - a_last).count();
    check(b.ok, "once A is silent past it, B, valid, takes the worker over");
    check(a_quiet_ms >= 1000 && a_quiet_ms < 4000, "... after A's 1 s without a message (not its keepalive or timeouts)");
    check(b.cl.reset({}, err) && b.cl.run(1, &tok1, 0, rows1, kHbf, back1, kHbf, err) &&
              std::memcmp(rows1, back1, sizeof rows1) == 0,
          "... and B is served");
    for (int i = 0; i < 100 && disconnects.load() == disc0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() == disc0 + 1, "A ended as a closed connection does (the disconnect step ran once)");
    err.clear();
    check(!a.reset({}, err) && stage_link_lost(err), "A's next message fails as a lost link");
    check(b.cl.reset({}, err), "... B is still served");
    // a served connection that closes: the one waiting is served at once, not after the idle time
    Pending c3;
    connect_later(c3, with_token(hello_of(), "tok"), wk3.where);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const bool b_last = b.cl.reset({}, err);
    const auto b_closed = SteadyClock::now();
    b.cl.close();
    for (int i = 0; i < 500 && !c3.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    c3.t.join();
    const double c_ms = std::chrono::duration<double, std::milli>(c3.at - b_closed).count();
    check(b_last && c3.ok && c_ms < 900, "B closes: the main waiting is served at once (under the 1 s idle time)");
    check(c3.cl.reset({}, err), "... and its messages are answered");
    a.close();
    c3.cl.close();
    bad.close();
    wk3.stop();
    err.clear();
    check(!greet_with(a, wk3.where, with_token(hello_of(), "tok")) && contains(err, "cannot connect") &&
              stage_link_lost(err),
          "nothing listening: a lost link (worth trying again)");
    a.close();
    err.clear();
    check(!greet_with(a, "name.invalid:5000", with_token(hello_of(), "tok")) && contains(err, "cannot resolve") &&
              stage_link_lost(err),
          "a name that does not resolve: a lost link (a name service may answer later)");
    a.close();
    check(!greet_with(a, "no-port", with_token(hello_of(), "tok")) && !stage_link_lost(err),
          "an address without a port: not a lost link");
    a.close();
    auto ms_between = [](SteadyClock::time_point from, SteadyClock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };

    // 9. STRATA_REMOTE_TAKEOVER_S=0: never taken over, however long the served one is silent
    set_takeover("0");
    Worker wk4;
    const bool wk4_up = wk4.start(h, buf, "tok");
    set_takeover("1");
    if (!wk4_up) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    check(greet_with(a, wk4.where, with_token(hello_of(), "tok")) && a.reset({}, err),
          "STRATA_REMOTE_TAKEOVER_S=0: main A is served");
    int disc = disconnects.load();
    Pending b4;
    connect_later(b4, with_token(hello_of(), "tok"), wk4.where);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    err.clear();
    check(!b4.done.load() && disconnects.load() == disc && a.reset({}, err),
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
    check(greet_with(a, wk5.where, with_token(hello_of(), "tok")) && a.reset({}, err), "main A is served");
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
    const bool a5_ok = a.reset({}, err);
    const double a5_ms = ms_between(a5_sent, SteadyClock::now());
    for (int i = 0; i < 300 && !quiet.closed.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(a5_ok && a5_ms < 1000, "A sends again: it is answered at once (the peer's part of a hello not waited for)");
    check(quiet.closed.load() && ms_between(a5_sent, quiet.at) < 2000,
          "... and the peer accepted meanwhile is closed then (not at its hello's 10 s)");
    quiet.send_part(hello_bytes, 8, hello_bytes.size());   // the rest of its hello, too late
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    err.clear();
    check(a.reset({}, err) && disconnects.load() == disc,
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
    check(greet_with(a, wk6.where, with_token(hello_of(), "tok")) && a.reset({}, err), "a relay's main A is served");
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
    a6_ok = a6_ok && a.reset({}, err);
    check(a6_ok && !b6.done.load() && disconnects.load() == disc,
          "a chunk's reply out after 1.5 s, A's next message 0.4 s later: A still served, B waits");
    for (int i = 0; i < 500 && !b6.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b6.t.join();
    check(b6.ok, "... B takes over once A is silent past 1 s");
    b6.cl.close();
    a.close();
    wk6.stop();

    // 10. alive(): the link between requests, checked with nothing sent and without blocking
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
    bool up = greet_with(a, wk7.where, with_token(hello_of(), "tok")) && a.reset({}, err);
    for (int i = 0; i < 20 && up; ++i) up = alive_timed(a);
    check(up && err.empty(), "an idle served link is alive (20 checks)");
    check(a.reset({}, err) && a.alive(err) && a.reset({}, err),
          "... the check sends nothing: the next messages and their replies are in step");
    Pending b7;
    connect_later(b7, with_token(hello_of(), "tok"), wk7.where);
    for (int i = 0; i < 500 && !b7.done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    b7.t.join();
    check(b7.ok, "main B takes the worker over once A is silent past 1 s");
    check(alive_until_down(a) && stage_link_lost(err),
          "A's check finds its connection ended, as a lost link (nothing sent)");
    check(!a.connected() && !a.alive(err) && stage_link_lost(err) && !a.reset({}, err) && stage_link_lost(err),
          "... and marks the link failed: later calls fail at once, as lost");
    check(b7.cl.alive(err) && b7.cl.reset({}, err), "B, served, is alive");
    wk7.stop();
    check(alive_until_down(b7.cl) && stage_link_lost(err), "the worker stopped: B's check finds the link lost");
    b7.cl.close();
    a.close();
    Chatty idle_peer, chatty;
    err.clear();
    up = idle_peer.start("") && greet_with(a, idle_peer.where, hello_of());
    for (int i = 0; i < 20 && up; ++i) {
        up = alive_timed(a);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    check(up, "a peer that sends nothing after its hello: alive (20 checks over 0.2 s)");
    a.close();
    idle_peer.stop();
    err.clear();
    check(chatty.start(std::string("\x01\x02\x03", 3)) && greet_with(a, chatty.where, hello_of()) &&
              alive_until_down(a) && contains(err, "out of step") && stage_link_lost(err) && !a.connected(),
          "a peer that sends bytes no message asked for: not alive, as lost (out of step)");
    a.close();
    chatty.stop();
    check(alive_worst_ms < 100, "no check above blocked (each under 100 ms)");

    // 11. connect() with a deadline (timeout_s) and a stop flag: the connect and the hello's reply together
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
    check(timed_connect(a, wk8.where, with_token(hello_of(), "tok"), 5, &never, ms) && a.reset({}, err) &&
              a.run(1, &tok1, 0, rows1, kHbf, back1, kHbf, err) && std::memcmp(rows1, back1, sizeof rows1) == 0,
          "a worker that answers: served within the deadline, the link blocking again for its messages");
    a.close();
    wk8.stop();

    // 12. h.idle (a relay's look at its next worker): only while the served main process is idle, about once a
    //     second; false ends that connection, which the main process's next message finds lost
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
    check(greet_with(a, wk9.where, with_token(hello_of(), "tok")) && a.reset({}, err), "a relay's main A is served");
    bool busy_ok = true;
    for (int i = 0; i < 6; ++i) {   // a message every 0.3 s, for 1.8 s
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        busy_ok = busy_ok && a.reset({}, err);
    }
    check(busy_ok && idle_looks.load() == 0, "no look while A sends a message every 0.3 s");
    std::this_thread::sleep_for(std::chrono::milliseconds(2600));
    const int looks = idle_looks.load();
    check(looks >= 1 && looks <= 3 && a.reset({}, err), "A idle for 2.6 s: about one look a second, and A still served");
    disc = disconnects.load();
    next_gone = true;
    const auto gone_at = SteadyClock::now();
    for (int i = 0; i < 300 && disconnects.load() == disc; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    check(disconnects.load() == disc + 1 && ms_between(gone_at, SteadyClock::now()) < 2500,
          "the next worker gone while A is idle: A's connection ends within the next look");
    err.clear();
    check(!a.reset({}, err) && stage_link_lost(err), "... A's next message finds it lost (its main process connects again)");
    next_gone = false;
    check(greet_with(a, wk9.where, with_token(hello_of(), "tok")) && a.reset({}, err), "... and connected again, A is served");
    a.close();
    wk9.stop();

    // 13. a worker reads nothing but a hello before the hello, and bounds a message's T before any product with it
    Worker wk10;
    if (!wk10.start(h, buf, "tok")) {
        std::printf("stage_link_test: no loopback port to listen on\n");
        return 1;
    }
    const int64_t wrap = (int64_t) 1 << 62;   // x a row's 4 floats x 4 bytes, and x 4 or 8 bytes a token: 0 in 64 bits
    auto refused_first = [&](StageMsg type, int64_t T, uint64_t bytes, const char* answer) {
        Raw p;
        if (!p.open(wk10.where)) return false;
        const auto t0 = SteadyClock::now();
        p.header(type, T, bytes);
        StageHeader r;
        std::string m;
        const bool answered = p.reply(r, m) && r.type == (uint32_t) StageMsg::Error && contains(m, answer);
        ms = ms_between(t0, SteadyClock::now());
        return answered && p.closed_by_worker();
    };
    auto serves_on = [&] {   // (a hello's reply waited for 5 s: a worker held fails here, not at the test's time-out)
        err.clear();
        a.close();
        const bool ok = a.connect(wk10.where, with_token(hello_of(), "tok"), peer, err, 5) &&
                        a.run(1, &tok1, 0, rows1, kHbf, back1, kHbf, err) &&
                        std::memcmp(rows1, back1, sizeof rows1) == 0;
        a.close();
        return ok;
    };
    check(refused_first(StageMsg::Run, wrap, 0, "hello first") && serves_on(),
          "a window of T = 2^62 before the hello: answered, closed, and the worker serves on");
    check(refused_first(StageMsg::Prefill, wrap, 0, "hello first") && serves_on(), "... a prompt chunk the same");
    check(refused_first(StageMsg::Run, 1, 4 + kHbf * 4, "hello first") && ms < 2000,
          "... a window of a good size: answered at once (its body, never sent, not waited for)");
    check(refused_first(StageMsg::Hello, 0, (uint64_t) 1 << 30, "bad hello") && ms < 2000 && serves_on(),
          "... a hello of a wrong size: answered at once (nothing drained)");
    // a hello sent a byte a second: closed at the hello's 10 s in all, not held while bytes keep coming
    {
        Silent drip;
        const auto t0 = SteadyClock::now();
        bool ok = drip.open(wk10.where);
        for (size_t i = 0; ok && i < hello_bytes.size() && !drip.closed.load(); ++i) {
            if (ms_between(t0, SteadyClock::now()) >= 13000) break;
            drip.send_part(hello_bytes, i, i + 1);
            for (int k = 0; k < 10 && !drip.closed.load(); ++k)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ok = ok && drip.closed.load();
        const double at = ok ? ms_between(t0, drip.at) : 0;
        drip.close();
        check(ok && at >= 9000 && at < 12000 && serves_on(),
              "a hello sent a byte a second: closed at 10 s in all, and the worker serves on");
    }
    // a hello with a wrong token, and headers right behind it: a window of T = 2^62, and one of a good size whose body
    // never comes.  Nothing after a hello is read before its token is taken, so neither ends the worker nor holds it
    // (a recv on that body is not woken by shutting the socket down on Windows).  The peer stays open meanwhile: its
    // close would end such a recv.  The refusal's text is not checked: closing with those headers unread resets the
    // connection, and the reset may drop it.
    std::vector<char> wrong(hello_bytes);
    {
        const StageHello nope = with_token(hello_of(), "nope");
        std::memcpy(wrong.data() + sizeof(StageHeader), &nope, sizeof nope);
    }
    {
        Raw p;
        const bool ok = p.open(wk10.where);
        p.send(wrong.data(), wrong.size());
        p.header(StageMsg::Run, wrap, 0);
        p.header(StageMsg::Run, 1, 4 + kHbf * 4);
        check(ok && p.closed_by_worker() && serves_on(),
              "a wrong token's hello, then a window of T = 2^62 and one whose body never comes: closed, the worker "
              "serves on");
    }
    // the same behind a second hello on a served connection: its token is taken before anything after it is read
    {
        Raw p;
        bool ok = p.open(wk10.where);
        p.send(hello_bytes.data(), hello_bytes.size());
        StageHeader r;
        std::string m;
        ok = ok && p.reply(r, m) && r.type == (uint32_t) StageMsg::HelloOk;
        p.send(wrong.data(), wrong.size());
        p.header(StageMsg::Run, wrap, 0);
        p.header(StageMsg::Run, 1, 4 + kHbf * 4);
        check(ok && p.closed_by_worker() && serves_on(),
              "... behind a second hello, a wrong token's, on a served connection: closed, the worker serves on");
    }
    // after the hello: the same window and chunk are malformed messages, and the link stays in step
    {
        Raw p;
        bool ok = p.open(wk10.where);
        p.send(hello_bytes.data(), hello_bytes.size());
        StageHeader r;
        std::string m;
        ok = ok && p.reply(r, m) && r.type == (uint32_t) StageMsg::HelloOk;
        p.header(StageMsg::Run, wrap, 0);
        ok = ok && p.reply(r, m) && r.type == (uint32_t) StageMsg::Error && contains(m, "bad window size");
        p.header(StageMsg::Prefill, wrap, 0);
        ok = ok && p.reply(r, m) && r.type == (uint32_t) StageMsg::Error && contains(m, "bad prompt chunk size");
        p.header(StageMsg::Run, 1, 4 + kHbf * 4);
        p.send(&tok1, 4);
        p.send(rows1, sizeof rows1);
        ok = ok && p.reply(r, m) && r.type == (uint32_t) StageMsg::RunOk && m.size() == sizeof rows1 &&
             std::memcmp(m.data(), rows1, sizeof rows1) == 0;
        check(ok, "after the hello, a window and a chunk of T = 2^62: malformed, and the next window runs");
    }
    wk10.stop();

    // 14. a hello's reply is read whole within the deadline, and an error reply's size is not taken on trust
    {
        StageHeader ok_head;
        ok_head.type = (uint32_t) StageMsg::HelloOk;
        ok_head.bytes = sizeof(StageHello);
        Chatty part;
        check(part.start(std::string((const char*) &ok_head, 8), false) &&
                  !timed_connect(a, part.where, hello_of(), 1, nullptr, ms) && ms >= 900 && ms < 3000 &&
                  contains(err, "timed out") && stage_link_lost(err),
              "a worker that sends 8 bytes of its hello's reply: given up at the 1 s, as a lost link");
        part.stop();
        Chatty part2;
        std::atomic<bool> stop_later{false};
        std::thread stopper2([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            stop_later = true;
        });
        const bool part2_ok = part2.start(std::string((const char*) &ok_head, 8), false) &&
                              timed_connect(a, part2.where, hello_of(), 30, &stop_later, ms);
        stopper2.join();
        check(!part2_ok && ms < 1500 && stage_wait_cancelled(err), "... a stop flag ends that wait too");
        part2.stop();
        StageHeader huge;
        huge.type = (uint32_t) StageMsg::Error;
        huge.bytes = (uint64_t) 1 << 62;
        Chatty liar;
        check(liar.start(std::string((const char*) &huge, sizeof huge), false) &&
                  !timed_connect(a, liar.where, hello_of(), 2, nullptr, ms) && ms < 1500 &&
                  contains(err, "unexpected reply to hello") && !stage_link_lost(err),
              "an error of 2^62 bytes as the hello's reply: not read, an unexpected reply (not a lost link)");
        liar.stop();
        Chatty liar2;
        err.clear();
        check(liar2.start(std::string((const char*) &huge, sizeof huge)) &&
                  greet_with(a, liar2.where, hello_of()) && !a.reset({}, err) && contains(err, "out of step") &&
                  stage_link_lost(err) && !a.connected(),
              "... as a later reply: not read, the link out of step (lost)");
        a.close();
        liar2.stop();
    }

    if (g_fail == 0) std::printf("stage_link_test: all passed\n");
    else std::printf("stage_link_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
