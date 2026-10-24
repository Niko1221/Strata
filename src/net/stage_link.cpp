// src/net/stage_link.cpp - see include/strata/net/stage_link.hpp.
#include "strata/net/stage_link.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
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
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <fcntl.h>
#endif

namespace strata::net {

// Seconds a read or write may block before the link counts as dead (STRATA_REMOTE_TIMEOUT_S; a prompt chunk on a
// slow worker can take several seconds, a first window after a long prompt a few more).
int io_timeout_s() {
    static const int s = [] {
        const char* v = std::getenv("STRATA_REMOTE_TIMEOUT_S");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n > 0 ? n : 300;
    }();
    return s;
}

namespace {

// Seconds a served connection must have had no message read, being read, or a reply going out before a new one may
// take the worker over (STRATA_REMOTE_TAKEOVER_S, default 30; 0: never).  A live main process can leave its worker
// waiting for seconds inside one request (its own layers of a prompt chunk, a long context), so a short pause is no
// sign it went away.
// Read when a worker starts listening (serve_stage).
int takeover_s() {
    const char* v = std::getenv("STRATA_REMOTE_TAKEOVER_S");
    return v != nullptr && *v != '\0' ? std::max(0, std::atoi(v)) : 30;
}

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// STRATA_REMOTE_TIMING=1: one line per prompt chunk and every 64th window, the worker's own time and the link's
bool remote_timing() {
    static const bool on = [] { const char* v = std::getenv("STRATA_REMOTE_TIMING"); return v && v[0] == '1'; }();
    return on;
}

// ---- the sockets of both systems behind one small set of calls
#if defined(_WIN32)
using sock_t = SOCKET;
const sock_t kNoSock = INVALID_SOCKET;
using sock_len = int;
using poll_fd = WSAPOLLFD;
constexpr int kSendFlags = 0;
// Winsock's SO_REUSEADDR binds over a port another process is listening on (two workers on one port, either one
// taking the connection); SO_EXCLUSIVEADDRUSE refuses that, as Linux does
constexpr int kListenOpt = SO_EXCLUSIVEADDRUSE;
int sock_close(sock_t s) { return ::closesocket(s); }
int sock_shutdown(sock_t s) { return ::shutdown(s, SD_BOTH); }
int sock_errno() { return WSAGetLastError(); }
bool sock_eintr(int e) { return e == WSAEINTR; }
bool sock_timedout(int e) { return e == WSAETIMEDOUT || e == WSAEWOULDBLOCK; }
bool sock_wouldblock(int e) { return e == WSAEWOULDBLOCK; }
int sock_poll(poll_fd* p, int n, int ms) { return ::WSAPoll(p, (ULONG) n, ms); }
std::string sock_errstr(int e) {
    char buf[256] = {};
    // English where the system has it: the log is read back as UTF-8 (stage_node), a localized ANSI text is not
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, (DWORD) e,
                             MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), buf, sizeof buf, nullptr);
    if (n == 0)
        n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, (DWORD) e, 0, buf,
                           sizeof buf, nullptr);
    std::string m(buf, n);
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' || m.back() == '.')) m.pop_back();
    return m + " (" + std::to_string(e) + ")";
}
/// a read's or a write's timeout (0: none) - Winsock takes milliseconds
void set_timeout(sock_t s, int opt, int seconds) {
    const DWORD ms = (DWORD) seconds * 1000;
    (void) setsockopt(s, SOL_SOCKET, opt, (const char*) &ms, sizeof ms);
}
/// Winsock needs WSAStartup once per process
bool net_init() {
    static const bool ok = [] {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    return ok;
}
#else
using sock_t = int;
const sock_t kNoSock = -1;
using sock_len = socklen_t;
using poll_fd = pollfd;
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;   // a closed peer is an error, not SIGPIPE
#else
constexpr int kSendFlags = 0;              // macOS: SO_NOSIGPIPE on each socket instead (tune_socket)
#endif
constexpr int kListenOpt = SO_REUSEADDR;   // a restart binds over its own TIME_WAIT, never over a live listener
int sock_close(sock_t s) { return ::close(s); }
int sock_shutdown(sock_t s) { return ::shutdown(s, SHUT_RDWR); }
int sock_errno() { return errno; }
bool sock_eintr(int e) { return e == EINTR; }
bool sock_timedout(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
bool sock_wouldblock(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
int sock_poll(poll_fd* p, int n, int ms) { return ::poll(p, (nfds_t) n, ms); }
std::string sock_errstr(int e) { return std::strerror(e); }
void set_timeout(sock_t s, int opt, int seconds) {
    timeval tv{};
    tv.tv_sec = seconds;
    (void) setsockopt(s, SOL_SOCKET, opt, &tv, sizeof tv);
}
bool net_init() { return true; }
#endif

template <class T>
void set_opt(sock_t s, int level, int name, T v) {
    (void) setsockopt(s, level, name, (const char*) &v, (sock_len) sizeof v);
}

sock_t S(std::intptr_t fd) { return (sock_t) fd; }

#if defined(_WIN32)
const char* gai_err(int rc) { return gai_strerrorA(rc); }   // the narrow one whatever UNICODE says
#else
const char* gai_err(int rc) { return gai_strerror(rc); }
#endif

void tune_socket(sock_t fd) {
    set_opt<int>(fd, IPPROTO_TCP, TCP_NODELAY, 1);
    set_opt<int>(fd, SOL_SOCKET, SO_KEEPALIVE, 1);
#if defined(SO_NOSIGPIPE)
    set_opt<int>(fd, SOL_SOCKET, SO_NOSIGPIPE, 1);   // macOS (no MSG_NOSIGNAL there): a closed peer is an error, not SIGPIPE
#endif
    // a peer that vanished without a FIN (power, cable) is noticed in about a minute, not the default two hours
    // (seconds on both systems; Windows 10 1709 and later)
#if defined(TCP_KEEPIDLE)
    set_opt<int>(fd, IPPROTO_TCP, TCP_KEEPIDLE, 30);
#endif
#if defined(TCP_KEEPINTVL)
    set_opt<int>(fd, IPPROTO_TCP, TCP_KEEPINTVL, 10);
#endif
#if defined(TCP_KEEPCNT)
    set_opt<int>(fd, IPPROTO_TCP, TCP_KEEPCNT, 3);
#endif
    set_opt<int>(fd, SOL_SOCKET, SO_SNDBUF, 8 << 20);
    set_opt<int>(fd, SOL_SOCKET, SO_RCVBUF, 8 << 20);
    set_timeout(fd, SO_RCVTIMEO, io_timeout_s());
    set_timeout(fd, SO_SNDTIMEO, io_timeout_s());
}

bool send_all(sock_t fd, const void* p, size_t n, std::string& err) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int k = (int) std::min<size_t>(n, (size_t) 1 << 30);   // Winsock counts in int
        const long long w = ::send(fd, c, k, kSendFlags);
        if (w < 0) {
            const int e = sock_errno();
            if (sock_eintr(e)) continue;
            err = std::string("remote stage: send failed: ") + sock_errstr(e);
            return false;
        }
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool recv_all(sock_t fd, void* p, size_t n, std::string& err) {
    char* c = (char*) p;
    while (n > 0) {
        const int k = (int) std::min<size_t>(n, (size_t) 1 << 30);
        const long long r = ::recv(fd, c, k, 0);
        if (r == 0) { err = "remote stage: the peer closed the connection"; return false; }
        if (r < 0) {
            const int e = sock_errno();
            if (sock_eintr(e)) continue;
            err = std::string("remote stage: receive failed: ") + (sock_timedout(e) ? std::string("timed out") : sock_errstr(e));
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// header + up to two payload parts, as one logical message
bool send_msg(sock_t fd, const StageHeader& h, const void* p1, size_t n1, const void* p2, size_t n2, std::string& err) {
    return send_all(fd, &h, sizeof h, err) && (n1 == 0 || send_all(fd, p1, n1, err)) &&
           (n2 == 0 || send_all(fd, p2, n2, err));
}

bool recv_header(sock_t fd, StageHeader& h, std::string& err) {
    if (!recv_all(fd, &h, sizeof h, err)) return false;
    if (h.magic != kStageMagic) { err = "remote stage: bad message magic (not a Strata stage peer?)"; return false; }
    return true;
}

bool drain(sock_t fd, uint64_t n, std::string& err) {
    char tmp[65536];
    while (n > 0) {
        const size_t k = (size_t) std::min<uint64_t>(n, sizeof tmp);
        if (!recv_all(fd, tmp, k, err)) return false;
        n -= k;
    }
    return true;
}

// ---- StageClient::connect with a deadline or a stop flag: the connect and the hello's reply are looked at every
// quarter second instead of blocking for the system's time-outs
constexpr const char* kCancelled = "remote stage: the wait was cancelled";
constexpr int kGaveUp = -1, kStopped = -2;   // connect_within: the deadline passed / `stop` was set

void set_blocking(sock_t s, bool on) {
#if defined(_WIN32)
    u_long nb = on ? 0 : 1;
    (void) ioctlsocket(s, FIONBIO, &nb);
#else
    const int f = fcntl(s, F_GETFL, 0);
    if (f >= 0) (void) fcntl(s, F_SETFL, on ? (f & ~O_NONBLOCK) : (f | O_NONBLOCK));
#endif
}

// the milliseconds of the next wait: a quarter second, or what is left before `deadline` (0: none left)
int slice_ms(bool timed, Clock::time_point deadline) {
    if (!timed) return 250;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return (int) std::clamp<long long>(left, 0, 250);
}

// 0 once connected, else the error number, kGaveUp or kStopped.  Without a deadline or a stop flag: the blocking
// connect, as always.
int connect_within(sock_t s, const sockaddr* a, sock_len n, bool timed, Clock::time_point deadline,
                   const std::atomic<bool>* stop) {
    if (!timed && stop == nullptr) return ::connect(s, a, n) == 0 ? 0 : sock_errno();
    set_blocking(s, false);
    if (::connect(s, a, n) != 0) {
        const int e = sock_errno();
#if defined(_WIN32)
        if (e != WSAEWOULDBLOCK) return e;
#else
        if (e != EINPROGRESS) return e;
#endif
        for (;;) {
            if (stop != nullptr && stop->load()) return kStopped;
            const int ms = slice_ms(timed, deadline);
            if (timed && ms == 0) return kGaveUp;
#if defined(_WIN32)
            // select, not WSAPoll: WSAPoll does not report a failed connect before Windows 10 2004
            fd_set w, x;
            FD_ZERO(&w);
            FD_ZERO(&x);
            FD_SET(s, &w);
            FD_SET(s, &x);
            timeval tv{ms / 1000, (ms % 1000) * 1000};
            const int r = ::select(0, nullptr, &w, &x, &tv);
#else
            poll_fd p{};
            p.fd = s;
            p.events = POLLOUT;
            const int r = sock_poll(&p, 1, ms);
#endif
            if (r > 0) break;   // connected or failed: SO_ERROR says which
            if (r < 0 && !sock_eintr(sock_errno())) return sock_errno();
        }
        int so = 0;
        sock_len sl = (sock_len) sizeof so;
        if (getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &so, &sl) != 0) return sock_errno();
        if (so != 0) return so;
    }
    set_blocking(s, true);
    return 0;
}

// until `s` has something to read (data, its end, an error: the read that follows says which): true; false with the
// reason when the deadline passes or `stop` is set first
bool wait_readable(sock_t s, Clock::time_point deadline, const std::atomic<bool>* stop, std::string& err) {
    for (;;) {
        if (stop != nullptr && stop->load()) { err = kCancelled; return false; }
        const int ms = slice_ms(true, deadline);
        if (ms == 0) { err = "remote stage: receive failed: timed out (no reply to the hello)"; return false; }
        poll_fd p{};
        p.fd = s;
        p.events = POLLIN;
        const int r = sock_poll(&p, 1, ms);
        if (r > 0 || (r < 0 && !sock_eintr(sock_errno()))) return true;
    }
}

// recv_all within the same deadline and stop flag: each part is waited for, then read as far as it is there (a recv
// on a readable socket does not block), so a peer that sends part of its reply and then nothing is given up at the
// deadline as one that sends nothing is
bool recv_within(sock_t s, void* p, size_t n, Clock::time_point deadline, const std::atomic<bool>* stop,
                 std::string& err) {
    char* c = (char*) p;
    while (n > 0) {
        if (!wait_readable(s, deadline, stop, err)) return false;
        const long long r = ::recv(s, c, (int) std::min<size_t>(n, (size_t) 1 << 30), 0);
        if (r == 0) { err = "remote stage: the peer closed the connection"; return false; }
        if (r < 0) {
            const int e = sock_errno();
            if (sock_eintr(e)) continue;
            err = std::string("remote stage: receive failed: ") + (sock_timedout(e) ? std::string("timed out") : sock_errstr(e));
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// the longest error text a peer's Error message is read for: a worker's errors are sentences; a longer one is not
// read (its size is not allocated) - the link is out of step, or the peer is not a Strata stage worker
constexpr uint64_t kMaxErrorBytes = 65536;

// the link failures a StageClient reports (stage_link_lost: what connecting again can mend; a name the name service
// cannot answer yet, after a reboot, is one)
constexpr const char* kLost[] = {"remote stage: cannot resolve ", "remote stage: cannot connect to ",
                                 "remote stage: the peer closed the connection",
                                 "remote stage: send failed", "remote stage: receive failed",
                                 "remote stage: the link failed earlier", "remote stage: not connected"};

bool hello_matches(const StageHello& a, const StageHello& b, std::string& err) {
    char buf[512];
    if (a.protocol != b.protocol || a.n_embd != b.n_embd || a.hc != b.hc || a.n_layers != b.n_layers ||
        a.n_expert != b.n_expert || a.layer_begin != b.layer_begin || a.handoff_floats != b.handoff_floats ||
        a.kv_type != b.kv_type || a.pack_hash != b.pack_hash) {
        std::snprintf(buf, sizeof buf,
                      "remote stage: the two sides differ (protocol %u/%u, n_embd %d/%d, hc %d/%d, layers %d/%d, "
                      "experts %d/%d, first remote layer %d/%d, hand-off floats %lld/%lld, K/V type %d/%d, "
                      "model pack %016llx/%016llx)",
                      a.protocol, b.protocol, a.n_embd, b.n_embd, a.hc, b.hc, a.n_layers, b.n_layers, a.n_expert,
                      b.n_expert, a.layer_begin, b.layer_begin, (long long) a.handoff_floats,
                      (long long) b.handoff_floats, a.kv_type, b.kv_type, (unsigned long long) a.pack_hash,
                      (unsigned long long) b.pack_hash);
        err = buf;
        return false;
    }
    return true;
}

// a main process's hello asks this worker to keep at most kStageCkptMax checkpoint parts: empty, else the refusal
std::string ckpt_max_refusal(const StageHello& h) {
    if (h.ckpt_max >= 0 && h.ckpt_max <= kStageCkptMax) return {};
    return "the main process keeps " + std::to_string(h.ckpt_max) + " conversation checkpoints, a worker at most " +
           std::to_string(kStageCkptMax) + " (--prompt-cache " + std::to_string(kStageCkptMax - 1) + " or less)";
}

}  // namespace

bool stage_link_lost(const std::string& err) {
    for (const char* p : kLost)
        if (err.rfind(p, 0) == 0) return true;
    return false;
}

bool stage_wait_cancelled(const std::string& err) { return err == kCancelled; }

// ------------------------------------------------------------------------------------------------ StageClient

StageClient::~StageClient() { close(); }

void StageClient::close() {
    if (fd_ >= 0) (void) sock_close(S(fd_));
    fd_ = -1;
    broken_ = false;
}

void StageClient::break_() {
    if (fd_ >= 0) (void) sock_shutdown(S(fd_));
    broken_ = true;
}

void StageClient::abort() {
    if (fd_ >= 0) (void) sock_shutdown(S(fd_));   // the waiting read fails and breaks the link itself
}

bool StageClient::connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err,
                          int timeout_s, const std::atomic<bool>* stop) {
    const bool timed = timeout_s > 0;
    const Clock::time_point deadline = Clock::now() + std::chrono::seconds(std::max(timeout_s, 0));
    // (a try that fails at once - a name that does not resolve yet, a network that is unreachable - never reaches the
    // polls below, so a stop flag that is already set is looked at here: a caller trying again every few seconds
    // stops at its next try instead of spinning through tries until its time is up)
    if (stop != nullptr && stop->load()) { err = kCancelled; return false; }
    if (!net_init()) { err = "remote stage: the sockets did not start (WSAStartup)"; return false; }
    const size_t colon = host_port.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= host_port.size()) {
        err = "remote stage: expected HOST:PORT, got \"" + host_port + "\"";
        return false;
    }
    const std::string host = host_port.substr(0, colon), port = host_port.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "remote stage: cannot resolve " + host_port + ": " + gai_err(rc);
        return false;
    }
    sock_t fd = kNoSock;
    int last = 0;
    for (addrinfo* ai = res; ai != nullptr && fd == kNoSock && last != kStopped; ai = ai->ai_next) {
        // (kGaveUp comes with a deadline only; slice_ms counts whole milliseconds, so less than one left is none)
        if (last == kGaveUp && Clock::now() + std::chrono::milliseconds(1) > deadline) break;
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == kNoSock) { last = sock_errno(); continue; }
        // an address that does not answer leaves the ones after it the rest of the deadline: 10 s at most each but
        // the last (a name whose first address has no route - an IPv6 one - while the next one works)
        const Clock::time_point by =
            ai->ai_next != nullptr ? std::min(deadline, Clock::now() + std::chrono::seconds(10)) : deadline;
        if ((last = connect_within(fd, ai->ai_addr, (sock_len) ai->ai_addrlen, timed, by, stop)) != 0) {
            (void) sock_close(fd);
            fd = kNoSock;
        }
    }
    freeaddrinfo(res);
    if (last == kStopped) { err = kCancelled; return false; }
    if (fd == kNoSock) {
        err = "remote stage: cannot connect to " + host_port + ": " + (last == kGaveUp ? "timed out" : sock_errstr(last));
        return false;
    }
    tune_socket(fd);
    fd_ = (std::intptr_t) fd;
    peer_ = host_port;
    {
        std::lock_guard<std::mutex> lq(q_mu_);
        sent_at_.clear();   // (a connection before this one: no chunk of it is outstanding here)
    }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Hello;
    h.bytes = sizeof mine;
    StageHeader r;
    // (the reply waits while the worker serves another main process: a take-over waits for its idle time.)  With a
    // stop flag but no deadline the reply is polled too, within STRATA_REMOTE_TIMEOUT_S as the blocking read would be.
    // The whole reply is read within it, not only its first byte.
    const bool polled = timed || stop != nullptr;
    const Clock::time_point reply_by = timed ? deadline : Clock::now() + std::chrono::seconds(io_timeout_s());
    auto take = [&](void* p, size_t n, std::string& e) {
        return polled ? recv_within(S(fd_), p, n, reply_by, stop, e) : recv_all(S(fd_), p, n, e);
    };
    if (!send_msg(S(fd_), h, &mine, sizeof mine, nullptr, 0, err) || !take(&r, sizeof r, err)) {
        close();
        return false;
    }
    if (r.magic != kStageMagic) {
        err = "remote stage: bad message magic (not a Strata stage peer?)";
        close();
        return false;
    }
    if (r.type == (uint32_t) StageMsg::Error && r.bytes <= kMaxErrorBytes) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) take(m.data(), m.size(), e2);
        err = "remote stage: the worker refused: " + m;
        close();
        return false;
    }
    if (r.type != (uint32_t) StageMsg::HelloOk || r.bytes != sizeof peer) {
        err = "remote stage: unexpected reply to hello (type " + std::to_string(r.type) + ", " +
              std::to_string(r.bytes) + " bytes)";
        close();
        return false;
    }
    if (!take(&peer, sizeof peer, err)) {
        close();
        return false;
    }
    if (!hello_matches(mine, peer, err)) { close(); return false; }
    if (mine.vision != 0 && peer.vision == 0) {   // (a worker of this protocol refuses such a hello itself)
        err = "remote stage: the worker was started without --vision (its layers need the image positions too)";
        close();
        return false;
    }
    return true;
}

bool StageClient::alive(std::string& err) {
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (broken_) { err = "remote stage: the link failed earlier"; return false; }
    if (fd_ < 0) { err = "remote stage: not connected"; return false; }
    for (;;) {
        poll_fd pfd{};
        pfd.fd = S(fd_);
        pfd.events = POLLIN;
        const int pr = sock_poll(&pfd, 1, 0);
        if (pr == 0) return true;   // nothing to read: the link stands (a peer gone without a word: not yet seen)
        if (pr < 0) {
            const int e = sock_errno();
            if (sock_eintr(e)) continue;
            err = std::string("remote stage: receive failed: ") + sock_errstr(e);
            break_();
            return false;
        }
        // readable, closed or failed: the peek says which, and does not block (poll said it would not)
        char c = 0;
        const long long r = ::recv(S(fd_), &c, 1, MSG_PEEK);
        if (r == 0) {
            err = "remote stage: the peer closed the connection";
        } else if (r < 0) {
            const int e = sock_errno();
            if (sock_eintr(e)) continue;
            if (sock_wouldblock(e)) return true;   // (nothing there after all; a keepalive's time-out is a failure)
            err = std::string("remote stage: receive failed: ") + sock_errstr(e);
        } else {
            err = "remote stage: receive failed: data no message asked for (the link is out of step)";
        }
        break_();
        return false;
    }
}

// the next reply: `want` with exactly reply_bytes of payload, or the worker's error (the connection then stays)
bool StageClient::read_reply_(StageMsg want, void* reply, size_t reply_bytes, int64_t& worker_us, std::string& err) {
    StageHeader r;
    if (broken_) { err = "remote stage: the link failed earlier"; return false; }
    if (!recv_header(S(fd_), r, err)) { break_(); return false; }
    if (r.type == (uint32_t) StageMsg::Error && r.bytes > kMaxErrorBytes) {
        err = "remote stage: receive failed: an error reply of " + std::to_string(r.bytes) +
              " bytes (the link is out of step)";
        break_();
        return false;
    }
    if (r.type == (uint32_t) StageMsg::Error) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) recv_all(S(fd_), m.data(), m.size(), e2);
        err = "remote stage (worker): " + m;
        return false;
    }
    if (r.type != (uint32_t) want || r.bytes != reply_bytes) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "remote stage: unexpected reply (type %u, %llu bytes; wanted type %u, %zu bytes)",
                      r.type, (unsigned long long) r.bytes, (unsigned) want, reply_bytes);
        err = buf;
        break_();
        return false;
    }
    if (reply_bytes > 0 && !recv_all(S(fd_), reply, reply_bytes, err)) { break_(); return false; }
    worker_us = r.a;
    stats_.bytes_in += sizeof r + reply_bytes;
    return true;
}

bool StageClient::run(int T, const int32_t* tokens, int64_t pos0, const float* rows_in, size_t in_floats,
                      float* rows_out, size_t out_floats, std::string& err) {
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    const auto t0 = Clock::now();
    StageHeader h;
    h.type = (uint32_t) StageMsg::Run;
    h.a = T;
    h.b = pos0;
    h.bytes = (uint64_t) T * sizeof(int32_t) + in_floats * sizeof(float);
    if (!send_msg(S(fd_), h, tokens, (size_t) T * sizeof(int32_t), rows_in, in_floats * sizeof(float), err)) {
        break_();
        return false;
    }
    stats_.bytes_out += sizeof h + h.bytes;
    int64_t wus = 0;
    const bool ok = read_reply_(StageMsg::RunOk, rows_out, out_floats * sizeof(float), wus, err);
    const double ms = ms_since(t0);
    ++stats_.runs;
    stats_.ms_run += ms;
    stats_.ms_run_worker += (double) wus / 1000.0;
    if (remote_timing() && (stats_.runs & 63) == 0)
        std::fprintf(stderr, "strata remote: %llu windows: mean %.2f ms a round trip = worker %.2f + link %.2f "
                             "(last: T=%d at %lld, %.2f = %.2f + %.2f)\n",
                     (unsigned long long) stats_.runs, stats_.ms_run / (double) stats_.runs,
                     stats_.ms_run_worker / (double) stats_.runs,
                     (stats_.ms_run - stats_.ms_run_worker) / (double) stats_.runs, T, (long long) pos0, ms,
                     (double) wus / 1000.0, ms - (double) wus / 1000.0);
    return ok;
}

bool StageClient::commit(int n_keep, std::string& err) {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Commit;
    h.a = n_keep;
    if (!send_msg(S(fd_), h, nullptr, 0, nullptr, 0, err)) { break_(); return false; }
    stats_.bytes_out += sizeof h;
    ++stats_.commits;
    return true;
}

bool StageClient::prefill_send(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows_in,
                               size_t row_floats, int64_t skip, std::string& err) {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    skip = std::max<int64_t>(0, std::min<int64_t>(skip, T));
    StageHeader h;
    h.type = (uint32_t) StageMsg::Prefill;
    h.a = T;
    h.b = pos0;
    h.c = (flags & 0xff) | (skip << 8);
    const size_t in_bytes = (size_t) T * row_floats * sizeof(float);
    h.bytes = (uint64_t) T * sizeof(int64_t) + in_bytes;
    {
        std::lock_guard<std::mutex> lq(q_mu_);
        sent_at_.push_back(Clock::now());
    }
    if (!send_msg(S(fd_), h, tokens, (size_t) T * sizeof(int64_t), rows_in, in_bytes, err)) { break_(); return false; }
    stats_.bytes_out += sizeof h + h.bytes;
    return true;
}

bool StageClient::prefill_recv(float* rows_out, size_t row_floats, int64_t T, int64_t skip, std::string& err) {
    std::lock_guard<std::mutex> lk(recv_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    skip = std::max<int64_t>(0, std::min<int64_t>(skip, T));
    int64_t wus = 0;
    const bool ok = read_reply_(StageMsg::PrefillOk, rows_out + (size_t) skip * row_floats,
                                (size_t) (T - skip) * row_floats * sizeof(float), wus, err);
    Clock::time_point t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lq(q_mu_);
        if (!sent_at_.empty()) {
            t0 = sent_at_.front();
            sent_at_.pop_front();
        }
    }
    const double ms = ms_since(t0);
    ++stats_.prefills;
    stats_.ms_prefill += ms;
    if (remote_timing())
        std::fprintf(stderr, "strata remote: chunk T=%lld: %.0f ms from its send to its rows = worker %.0f + link and "
                             "queue %.0f (%lld rows back)\n",
                     (long long) T, ms, (double) wus / 1000.0, ms - (double) wus / 1000.0, (long long) (T - skip));
    return ok;
}

bool StageClient::keep_call_(StageMsg type, StageMsg want, int64_t a, int64_t b, const std::vector<int64_t>& keep,
                             int64_t& reply_a, std::string& err) {
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) type;
    h.a = a;
    h.b = b;
    h.bytes = keep.size() * sizeof(int64_t);   // (a Reset that keeps nothing is the header alone, as before checkpoints)
    if (!send_msg(S(fd_), h, keep.data(), (size_t) h.bytes, nullptr, 0, err)) { break_(); return false; }
    // a Reset is counted as before checkpoints (in `resets`, not in bytes_out); the checkpoint messages' bytes are
    if (type == StageMsg::Reset) ++stats_.resets;
    else stats_.bytes_out += sizeof h + h.bytes;
    return read_reply_(want, nullptr, 0, reply_a, err);
}

bool StageClient::reset(const std::vector<int64_t>& keep, std::string& err) {
    int64_t a = 0;
    return keep_call_(StageMsg::Reset, StageMsg::ResetOk, 0, 0, keep, a, err);
}

bool StageClient::ckpt_save(int64_t id, int64_t L, const std::vector<int64_t>& keep, bool& stored, std::string& err) {
    int64_t a = 0;
    stored = false;
    if (!keep_call_(StageMsg::CkptSave, StageMsg::CkptSaveOk, id, L, keep, a, err)) return false;
    stored = a == 1;
    return true;
}

bool StageClient::ckpt_restore(int64_t id, int64_t L, const std::vector<int64_t>& keep, bool& missing,
                               std::string& err) {
    int64_t a = 0;
    missing = false;
    if (keep_call_(StageMsg::CkptRestore, StageMsg::CkptRestoreOk, id, L, keep, a, err)) return true;
    // the worker's own text (read_reply_'s prefix, then the fixed start) - a relay's forwarded error is not this
    missing = connected() && err.rfind(std::string("remote stage (worker): ") + kStageCkptMissing, 0) == 0;
    return false;
}

bool StageClient::positions(const int32_t* table, int64_t cells, std::string& err) {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Positions;
    h.a = table != nullptr ? cells : 0;
    h.b = table != nullptr ? 0 : 1;
    h.bytes = table != nullptr ? (uint64_t) cells * 3 * sizeof(int32_t) : 0;
    if (table != nullptr && cells < 1) { err = "remote stage: an empty position table"; return false; }
    if (!send_msg(S(fd_), h, table, (size_t) h.bytes, nullptr, 0, err)) { break_(); return false; }
    stats_.bytes_out += sizeof h + h.bytes;
    return true;
}

// ------------------------------------------------------------------------------------------------ serve_stage

namespace {

// One message as the receiving thread read it.  A prompt chunk's rows are already in buf.pf_in[slot].
struct Inbox {
    StageHeader h;
    std::vector<int32_t> tok32;
    std::vector<int64_t> tok64;
    std::vector<float> rows;    ///< a window's rows (copied to run_in by the serving thread)
    std::vector<int64_t> keep;  ///< Reset / CkptSave / CkptRestore: the checkpoint ids the main process holds
    StageHello hello;
    int slot = -1;
    std::string bad;            ///< the message was malformed: reply this error
    bool closed = false;        ///< the connection ended (err says why)
    std::string err;
};

// A main process's connection: tuned, its hello due within 10 s (a silent peer does not hold the worker; after the
// hello there is no read timeout - the main process may wait for its next prompt for hours), logged.
sock_t accept_main(sock_t lfd, std::string& host) {
    sockaddr_storage peer{};
    sock_len plen = sizeof peer;
    const sock_t fd = ::accept(lfd, (sockaddr*) &peer, &plen);
    if (fd == kNoSock) return fd;
    char name[128] = "?";
    (void) getnameinfo((sockaddr*) &peer, plen, name, sizeof name, nullptr, 0, NI_NUMERICHOST);
    host = name;
    tune_socket(fd);
    set_timeout(fd, SO_RCVTIMEO, 10);
    std::fprintf(stderr, "strata stage worker: connection from %s\n", name);
    std::fflush(stderr);
    return fd;
}

// The first message of a connection (one that arrived while another one is served, or the one served now): a hello
// read whole (its read timeout lifted), else `bad` or `closed`.  Nothing else is read: such a connection is refused
// and closed.  Every read is polled, within the hello's 10 s in all, so `drop` ends it at once (shutting a socket
// down does not wake a recv blocked on it on Windows) - also when the peer sent part of a hello, or sends it a byte
// at a time.
Inbox read_hello(sock_t fd, const std::atomic<bool>& drop) {
    Inbox in;
    const Clock::time_point t0 = Clock::now();
    auto take = [&](void* p, size_t n) -> bool {
        char* c = (char*) p;
        while (n > 0) {
            bool ready = false;
            while (!ready) {
                if (drop.load()) return false;
                if (ms_since(t0) >= 10000.0) { in.err = "remote stage: receive failed: timed out"; return false; }
                poll_fd pfd{};
                pfd.fd = fd;
                pfd.events = POLLIN;
                const int pr = sock_poll(&pfd, 1, 50);
                if (pr < 0) {
                    const int e = sock_errno();
                    if (sock_eintr(e)) continue;
                    in.err = "remote stage: receive failed: " + sock_errstr(e);
                    return false;
                }
                ready = pr > 0;
            }
            const long long r = ::recv(fd, c, (int) n, 0);   // (what is there: it does not block)
            if (r == 0) { in.err = "remote stage: the peer closed the connection"; return false; }
            if (r < 0) {
                const int e = sock_errno();
                if (sock_eintr(e)) continue;
                in.err = "remote stage: receive failed: " + sock_errstr(e);
                return false;
            }
            c += r;
            n -= (size_t) r;
        }
        return true;
    };
    if (!take(&in.h, sizeof in.h)) {
        in.closed = true;
    } else if (in.h.magic != kStageMagic) {
        in.closed = true;
        in.err = "remote stage: bad message magic (not a Strata stage peer?)";
    } else if ((StageMsg) in.h.type != StageMsg::Hello) {
        in.bad = "hello first";
    } else if (in.h.bytes != sizeof in.hello) {
        in.bad = "bad hello";
    } else if (!take(&in.hello, sizeof in.hello)) {
        in.closed = true;
    } else {
        set_timeout(fd, SO_RCVTIMEO, 0);
    }
    return in;
}

void send_error(sock_t fd, const std::string& m) {
    StageHeader e;
    e.type = (uint32_t) StageMsg::Error;
    e.bytes = m.size();
    std::string se;
    (void) send_msg(fd, e, m.data(), m.size(), nullptr, 0, se);
}

}  // namespace

StageRelay::~StageRelay() {
    std::string e;
    (void) wait(e);
}

bool StageRelay::wait(std::string& err) {
    if (!out_.valid()) return true;
    if (out_.get()) return true;
    err = err_;
    return false;
}

bool StageRelay::send(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows,
                      size_t row_floats, int64_t skip, std::string& err) {
    if (!wait(err)) return false;
    tokens_.assign(tokens, tokens + T);
    err_.clear();
    out_ = std::async(std::launch::async, [this, T, pos0, flags, rows, row_floats, skip] {
        return next_.prefill_send(tokens_.data(), T, pos0, flags, rows, row_floats, skip, err_);
    });
    return true;
}

int serve_stage(const std::string& bind_addr, int port, const std::string& token, const StageHandlers& h,
                const StageBuffers& buf, StageStats& stats, const std::atomic<bool>* stop) {
    if (!net_init()) {
        std::fprintf(stderr, "strata stage worker: the sockets did not start (WSAStartup)\n");
        return 1;
    }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = bind_addr.empty() ? AF_INET6 : AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string port_s = std::to_string(port);
    if (const int rc = getaddrinfo(bind_addr.empty() ? nullptr : bind_addr.c_str(), port_s.c_str(), &hints, &res); rc != 0) {
        std::fprintf(stderr, "strata stage worker: cannot resolve %s: %s\n", bind_addr.c_str(), gai_err(rc));
        return 1;
    }
    const sock_t lfd = ::socket(res->ai_family, SOCK_STREAM, 0);
    if (lfd == kNoSock) {
        std::fprintf(stderr, "strata stage worker: socket: %s\n", sock_errstr(sock_errno()).c_str());
        freeaddrinfo(res);
        return 1;
    }
    set_opt<int>(lfd, SOL_SOCKET, kListenOpt, 1);
    if (res->ai_family == AF_INET6) set_opt<int>(lfd, IPPROTO_IPV6, IPV6_V6ONLY, 0);
    if (::bind(lfd, res->ai_addr, (sock_len) res->ai_addrlen) != 0 || ::listen(lfd, 1) != 0) {
        std::fprintf(stderr, "strata stage worker: cannot listen on %s:%d: %s\n", bind_addr.empty() ? "*" : bind_addr.c_str(),
                     port, sock_errstr(sock_errno()).c_str());
        (void) sock_close(lfd);
        freeaddrinfo(res);
        return 1;
    }
    freeaddrinfo(res);
    std::fprintf(stderr, "strata stage worker: listening on %s:%d%s\n", bind_addr.empty() ? "*" : bind_addr.c_str(), port,
                 token.empty() ? " - WARNING: no STRATA_STAGE_TOKEN, any host that reaches this port can drive the worker"
                               : " (a token is required)");
    std::fflush(stderr);
    // A connection that arrives while a main process is served (its PC lost power, say, and its process started
    // again) waits in the listen backlog until the served one has had no message read, being read, or a reply going
    // out for `takeover` seconds.  Then it is accepted, its hello read on a thread of its own, and a valid hello (the
    // token, the same checks as below) takes the worker over: the old connection ends as a closed one does, and the
    // new one is served.  An invalid one is refused; the old one stays.  One accepted while the served main process
    // sends again is closed (its main process connects again).
    const int takeover = takeover_s();
    sock_t cand = kNoSock;
    std::string cand_host;
    std::atomic<bool> cand_drop{false};   // ends the wait for its hello (drop_cand; declared first: it outlives it)
    std::future<Inbox> cand_hello;
    // the connection served next: accepted while another was served, its hello read (`first`)
    sock_t next_fd = kNoSock;
    std::string next_host;
    Inbox first;
    while (stop == nullptr || !*stop) {
        sock_t fd = next_fd;
        std::string host = next_host;
        const bool have_first = fd != kNoSock;
        next_fd = kNoSock;
        if (fd == kNoSock) {
            poll_fd pfd{};
            pfd.fd = lfd;
            pfd.events = POLLIN;
            const int pr = sock_poll(&pfd, 1, 1000);
            if (pr < 0 && !sock_eintr(sock_errno())) {   // (a network that went down: no spinning on the error)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            if (pr <= 0) continue;
            fd = accept_main(lfd, host);
            if (fd == kNoSock) continue;
        }

        // ---- the receiving thread: whole messages into a queue, a prompt chunk's rows straight into a free pf_in
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Inbox> q;
        if (have_first) q.push_back(std::move(first));   // (read with its connection: the reader starts after it)
        first = Inbox();
        bool slot_busy[2] = {false, false};
        std::atomic<bool> quit{false};
        std::atomic<bool> reading{false};   // a message's header is in, the rest not yet queued
        bool admitted = false;              // (under mu) the serving thread took this connection's hello: read on
        std::thread reader([&] {
            // The first message is read as a waiting connection's is (read_hello): a hello, whole within 10 s, and
            // nothing else.  A peer that sends it a byte at a time holds the worker no longer, and another message's
            // body is neither received nor drained (its sizes come from a peer no token has admitted): the serving
            // thread answers it and ends the connection.  After a hello (the first, or another a served main process
            // sends) nothing is read until the serving thread has admitted it (its token): a refused peer's next
            // header, with a body that never comes, would hold this thread in a recv that shutting the socket down
            // does not wake on Windows.  Then the wait for each message is polled, so this connection's end (quit)
            // ends it, for the same reason.
            bool hello_in = have_first;
            for (;;) {
                Inbox in;
                bool last = false;   // a first message that is not a whole hello: nothing after it is read
                const bool first_read = !hello_in;
                if (first_read) {
                    in = read_hello(fd, quit);
                    hello_in = !in.closed && in.bad.empty();
                    last = !hello_in;
                } else {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return admitted || quit.load(); });
                    }
                    bool ready = false;
                    while (!ready && !quit.load()) {
                        poll_fd pfd{};
                        pfd.fd = fd;
                        pfd.events = POLLIN;
                        ready = sock_poll(&pfd, 1, 250) != 0;   // (data, the peer's end or an error: the recv says)
                    }
                    if (!ready) break;
                    reading.store(true);   // (from here until it is queued: not idle)
                }
                if (first_read) {
                    // (read whole above)
                } else if (!recv_header(fd, in.h, in.err)) {
                    in.closed = true;
                } else {
                    const StageMsg type = (StageMsg) in.h.type;
                    if (type == StageMsg::Hello) {
                        if (in.h.bytes != sizeof in.hello) {
                            in.bad = "bad hello";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else if (!recv_all(fd, &in.hello, sizeof in.hello, in.err)) {
                            in.closed = true;
                        } else {
                            set_timeout(fd, SO_RCVTIMEO, 0);
                        }
                    } else if (type == StageMsg::Run) {
                        // T is bounded before any product with it: a T near 2^62 times a row's floats wraps around to
                        // a size that passes, and its resize would end the process
                        const int64_t T = in.h.a;
                        const int64_t max_t =
                            (int64_t) (buf.run_floats / (size_t) std::max<int64_t>(buf.handoff_floats, 1));
                        const size_t rows = T >= 1 && T <= max_t ? (size_t) T * (size_t) buf.handoff_floats : 0;
                        if (T < 1 || T > max_t || in.h.bytes != (uint64_t) T * 4 + rows * 4) {
                            in.bad = "bad window size";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else {
                            in.tok32.resize((size_t) T);
                            in.rows.resize(rows);
                            if (!recv_all(fd, in.tok32.data(), (size_t) T * 4, in.err) ||
                                !recv_all(fd, in.rows.data(), rows * 4, in.err))
                                in.closed = true;
                        }
                    } else if (type == StageMsg::Prefill) {
                        const int64_t T = in.h.a;   // (bounded first, as a window's)
                        const int64_t max_t =
                            (int64_t) (buf.pf_floats / (size_t) std::max<int64_t>(buf.pf_row_floats, 1));
                        const size_t rows = T >= 1 && T <= max_t ? (size_t) T * (size_t) buf.pf_row_floats : 0;
                        if (T < 1 || T > max_t || in.h.bytes != (uint64_t) T * 8 + rows * 4) {
                            in.bad = "bad prompt chunk size";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else {
                            {   // a free buffer: the serving thread releases one when it has read that chunk
                                std::unique_lock<std::mutex> lk(mu);
                                cv.wait(lk, [&] { return !slot_busy[0] || !slot_busy[1] || quit.load(); });
                                if (quit.load()) break;
                                in.slot = !slot_busy[0] ? 0 : 1;
                                slot_busy[in.slot] = true;
                            }
                            in.tok64.resize((size_t) T);
                            if (!recv_all(fd, in.tok64.data(), (size_t) T * 8, in.err) ||
                                !recv_all(fd, buf.pf_in[in.slot], rows * 4, in.err))
                                in.closed = true;
                        }
                    } else if (type == StageMsg::Positions) {
                        // (t, h, w) per cell; at most 4M cells (the handler checks it against the context)
                        const int64_t cells = in.h.a;
                        const bool identity = in.h.b == 1;
                        if (identity ? in.h.bytes != 0
                                     : (cells < 1 || cells > ((int64_t) 1 << 22) || in.h.bytes != (uint64_t) cells * 12)) {
                            in.bad = "bad position table size";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else if (!identity) {
                            in.tok32.resize((size_t) cells * 3);
                            if (!recv_all(fd, in.tok32.data(), (size_t) cells * 12, in.err)) in.closed = true;
                        }
                    } else if (type == StageMsg::Reset || type == StageMsg::CkptSave || type == StageMsg::CkptRestore) {
                        if (in.h.bytes % 8 != 0 || in.h.bytes / 8 > (uint64_t) kStageCkptMax) {
                            in.bad = "bad checkpoint list";
                            if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                        } else {
                            in.keep.resize((size_t) (in.h.bytes / 8));
                            if (!recv_all(fd, in.keep.data(), (size_t) in.h.bytes, in.err)) in.closed = true;
                        }
                    } else if (type != StageMsg::Commit) {
                        in.bad = "unknown message type " + std::to_string(in.h.type);
                        if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                    }
                }
                const bool closed = in.closed;
                // a whole hello (a served main process may send another): nothing after it is read until the serving
                // thread has taken it, as after the first
                const bool hello = !closed && in.bad.empty() && (StageMsg) in.h.type == StageMsg::Hello;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (hello) admitted = false;
                    q.push_back(std::move(in));
                    reading.store(false);
                }
                cv.notify_all();
                if (closed || last || quit.load()) break;
            }
        });

        // ---- the serving thread (this one): the messages in order
        // a failed one-way message, reported on the next reply: pending_err a Commit's (a Reset drops it: it was the
        // request before), pos_err the position table's (the request the next Reset starts: it is reported there; a
        // later table replaces it, as the main process sent no Reset for a request it refused)
        std::string err, pending_err, pos_err;
        // both pending errors, cleared: what the next Run or Prefill answers instead
        auto take_pending = [&]() {
            std::string e = pending_err.empty() ? pos_err : pos_err.empty() ? pending_err : pending_err + "; " + pos_err;
            pending_err.clear();
            pos_err.clear();
            return e;
        };
        bool greeted = false;
        // the position the session holds (a checkpoint is saved only there): a chunk's end, a window's first
        // position plus its accepted count, 0 after a reset, L after a checkpoint restore
        int64_t pos = 0, run_pos0 = 0;
        // a prompt chunk's reply goes out on its own thread while the next chunk is read; every later send waits
        // for it (the replies keep their order), and so does the reuse of its output buffer
        std::future<bool> reply_out;
        std::string reply_err;
        int out_buf = 0;
        auto replies_out = [&]() -> bool {
            if (!reply_out.valid()) return true;
            if (reply_out.get()) return true;
            err = reply_err;
            return false;
        };
        auto reply_error = [&](const std::string& m) -> bool {
            if (!replies_out()) return false;
            StageHeader e;
            e.type = (uint32_t) StageMsg::Error;
            e.bytes = m.size();
            std::string se;
            return send_msg(fd, e, m.data(), m.size(), nullptr, 0, se);
        };
        StageHello greeted_mine;   // this worker's hello to the main process served: a new one's is checked against it
        // a new connection's hello (read on its thread): whether it takes the worker over; else it is refused and
        // logged.  The hello handler runs again when the new connection is served (it checks and records only).
        auto takes_over = [&](const Inbox& ci) -> bool {
            std::string why;
            if (ci.closed) {
                why = ci.err.empty() ? "no hello" : ci.err;
            } else if (!ci.bad.empty()) {
                why = ci.bad;
                send_error(cand, why);
            } else {
                const std::string theirs_token(ci.hello.token, strnlen(ci.hello.token, sizeof ci.hello.token));
                StageHello mine;
                if (!token.empty() && theirs_token != token) {
                    why = "a wrong token";
                    send_error(cand, "wrong token (STRATA_STAGE_TOKEN)");
                } else if (!hello_matches(ci.hello, greeted_mine, why)) {
                    send_error(cand, why);
                } else if (!(why = ckpt_max_refusal(ci.hello)).empty() || !h.hello(ci.hello, mine, why)) {
                    // (the checkpoint limit before the handler: it records the new main process's values)
                    send_error(cand, why);
                } else {
                    return true;
                }
            }
            std::fprintf(stderr, "strata stage worker: %s refused (%s); %s is still served\n", cand_host.c_str(),
                         why.c_str(), host.c_str());
            std::fflush(stderr);
            return false;
        };
        // a connection accepted while the served main process was idle, which sent again before the new one's hello
        // was decided: the new one is closed without a word - its main process sees the connection closed (a lost
        // link) and connects again, waiting in the backlog until this one is idle once more.  (A held one would
        // otherwise take over at the next idle time with a hello read long before, from a peer that may be gone.)
        auto drop_cand = [&] {
            cand_drop = true;   // (a hello not in yet, or not whole, is not waited for)
            (void) sock_shutdown(cand);
            (void) cand_hello.get();
            (void) sock_close(cand);
            std::fprintf(stderr,
                         "strata stage worker: %s connected while %s was served: closed (it may connect again)\n",
                         cand_host.c_str(), host.c_str());
            std::fflush(stderr);
            cand = kNoSock;
        };
        // the next message (true); once the served main process has had no message read, being read, or a reply
        // going out for `takeover` seconds, a new connection that takes over (false: its hello in `first`).  Before
        // that a new connection stays in the listen backlog (its main process waits for its hello's reply).
        double idle_s = 0;   // (the take-over's log line)
        auto next_message = [&](Inbox& in) -> bool {
            Clock::time_point quiet_from = Clock::now();   // the last message is served
            Clock::time_point idle_looked = quiet_from;    // h.idle's last look
            for (;;) {
                bool reading_now;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    if (cv.wait_for(lk, std::chrono::milliseconds(cand != kNoSock ? 20 : 250), [&] { return !q.empty(); })) {
                        in = std::move(q.front());
                        q.pop_front();
                        lk.unlock();
                        if (cand != kNoSock && !in.closed) drop_cand();   // (a closed one: the held one is served next)
                        return true;
                    }
                    reading_now = reading.load();   // (with the queue: a message is either queued or being read)
                }
                if (stop != nullptr && *stop) {
                    in.closed = true;
                    in.err = "the worker is stopping";
                    return true;
                }
                // the idle time starts when the last message is read and its reply is out (a relay's reply can wait
                // on its next worker for long)
                if (reading_now ||
                    (reply_out.valid() && reply_out.wait_for(std::chrono::seconds(0)) != std::future_status::ready))
                    quiet_from = Clock::now();
                idle_s = ms_since(quiet_from) / 1000.0;
                // a relay's look at its next worker: one lost while idle ends this connection (h.idle)
                if (greeted && h.idle && idle_s >= 1.0 && ms_since(idle_looked) >= 1000.0) {
                    idle_looked = Clock::now();
                    std::string ie;
                    if (!h.idle(ie)) {
                        in.closed = true;
                        in.err = ie;
                        return true;
                    }
                }
                if (!greeted || takeover == 0 || idle_s < (double) takeover) continue;
                if (cand == kNoSock) {
                    poll_fd pfd{};
                    pfd.fd = lfd;
                    pfd.events = POLLIN;
                    if (sock_poll(&pfd, 1, 0) > 0 && (cand = accept_main(lfd, cand_host)) != kNoSock) {
                        cand_drop = false;
                        cand_hello = std::async(std::launch::async,
                                                [c = cand, &cand_drop] { return read_hello(c, cand_drop); });
                    }
                    continue;
                }
                if (cand_hello.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
                Inbox ci = cand_hello.get();
                if (takes_over(ci)) {
                    first = std::move(ci);
                    return false;
                }
                (void) sock_close(cand);
                cand = kNoSock;
            }
        };
        for (;;) {
            Inbox in;
            if (!next_message(in)) {
                char idle[64];
                std::snprintf(idle, sizeof idle, "%.0f s", idle_s);
                err = "a new main process (" + cand_host + ") took over after " + idle + " without a message";
                next_fd = cand;
                next_host = cand_host;
                cand = kNoSock;
                break;
            }
            if (in.closed) { err = in.err; break; }
            auto release = [&] {
                if (in.slot < 0) return;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    slot_busy[in.slot] = false;
                }
                in.slot = -1;
                cv.notify_all();
            };
            const StageMsg type = (StageMsg) in.h.type;
            // a first message that is not a whole hello ends the connection after its answer (its reader read nothing
            // after it, nor its body: the size it claims is not counted)
            if (!greeted && (type != StageMsg::Hello || !in.bad.empty())) {
                err = type != StageMsg::Hello ? "hello first" : in.bad;
                (void) reply_error(err);
                release();
                break;
            }
            stats.bytes_in += sizeof in.h + in.h.bytes;
            if (!in.bad.empty()) {
                release();
                // a one-way message gets no reply of its own: a malformed table is the next reply's error, as its
                // failure would be (an immediate Error would answer the next request in its place)
                if (type == StageMsg::Positions) {
                    pos_err = in.bad;
                    continue;
                }
                if (!reply_error(in.bad)) break;
                continue;
            }
            bool keep = true;
            const auto t0 = Clock::now();
            switch (type) {
            case StageMsg::Hello: {
                StageHello mine;
                std::string he;
                const std::string theirs_token(in.hello.token, strnlen(in.hello.token, sizeof in.hello.token));
                if (!token.empty() && theirs_token != token) {
                    std::fprintf(stderr, "strata stage worker: %s sent a wrong token: refused\n", host.c_str());
                    (void) reply_error("wrong token (STRATA_STAGE_TOKEN)");
                    keep = false;
                    break;
                }
                if (!h.hello(in.hello, mine, he) || !hello_matches(in.hello, mine, he)) {
                    (void) reply_error(he);
                    keep = false;
                    break;
                }
                // (after the protocol check: an older peer's hello has its build text here)
                if (const std::string r = ckpt_max_refusal(in.hello); !r.empty()) {
                    (void) reply_error(r);
                    keep = false;
                    break;
                }
                StageHeader r;
                r.type = (uint32_t) StageMsg::HelloOk;
                r.bytes = sizeof mine;
                if (!replies_out() || !send_msg(fd, r, &mine, sizeof mine, nullptr, 0, err)) keep = false;
                greeted = true;
                greeted_mine = mine;
                if (keep) {   // its reader reads on (a HelloOk that did not go out ends the connection)
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        admitted = true;
                    }
                    cv.notify_all();
                }
                std::fprintf(stderr, "strata stage worker: main process %s: layers %d.. here, context %lld, chunk %lld\n",
                             host.c_str(), in.hello.layer_begin, (long long) in.hello.max_context, (long long) in.hello.chunk);
                std::fflush(stderr);
                break;
            }
            case StageMsg::Run: {
                const int T = (int) in.h.a;
                std::memcpy(buf.run_in, in.rows.data(), in.rows.size() * 4);
                std::string e;
                bool ok;
                if (!pending_err.empty() || !pos_err.empty()) { e = take_pending(); ok = false; }
                else ok = h.run(T, in.tok32.data(), in.h.b, e);
                if (!ok) { keep = reply_error(e); break; }
                run_pos0 = in.h.b;
                StageHeader r;
                r.type = (uint32_t) StageMsg::RunOk;
                r.a = (int64_t) (ms_since(t0) * 1000.0);   // the worker's own time, microseconds
                r.bytes = in.rows.size() * 4;
                if (!replies_out() || !send_msg(fd, r, buf.run_out, in.rows.size() * 4, nullptr, 0, err)) keep = false;
                stats.bytes_out += sizeof r + r.bytes;
                ++stats.runs;
                stats.ms_run += ms_since(t0);
                break;
            }
            case StageMsg::Commit: {
                std::string e;
                if (pending_err.empty() && !h.commit((int) in.h.a, e)) pending_err = e;
                if (pending_err.empty()) pos = run_pos0 + in.h.a;
                ++stats.commits;
                break;
            }
            case StageMsg::Prefill: {
                const int64_t T = in.h.a;
                const int64_t skip = std::max<int64_t>(0, std::min<int64_t>(in.h.c >> 8, T));
                std::string e;
                bool ok;
                // this chunk's output buffer: the one whose reply went out two chunks ago - the reply in flight (the
                // last chunk's) uses the other
                float* const out = buf.pf_out[out_buf];
                if (!pending_err.empty() || !pos_err.empty()) { e = take_pending(); ok = false; }
                else ok = h.prefill(in.tok64.data(), T, in.h.b, in.h.c & 0xff, skip, buf.pf_in[in.slot], out, e);
                release();   // the chunk is read (its rows were uploaded before `prefill` returned)
                if (!ok) { keep = reply_error(e); break; }
                pos = in.h.b + T;
                const double work_ms = ms_since(t0);
                if (remote_timing())
                    std::fprintf(stderr, "strata stage worker: chunk T=%lld at %lld: read in %.0f ms\n", (long long) T,
                                 (long long) in.h.b, work_ms);
                const size_t back = (size_t) (T - skip) * (size_t) buf.pf_row_floats;
                if (!replies_out()) { keep = false; break; }   // the previous reply is out: its buffer is free next time
                StageHeader r;
                r.type = (uint32_t) StageMsg::PrefillOk;
                r.a = (int64_t) (work_ms * 1000.0);
                r.bytes = back * 4;
                const float* src = out + (size_t) skip * (size_t) buf.pf_row_floats;
                reply_err.clear();
                reply_out = std::async(std::launch::async, [fd, r, src, back, out, T, skip, &h, &reply_err] {
                    if (h.prefill_reply) {   // a relay worker: the next worker's rows (in chunk order)
                        std::string e;
                        if (!h.prefill_reply(out, T, skip, e)) {
                            StageHeader eh;
                            eh.type = (uint32_t) StageMsg::Error;
                            eh.bytes = e.size();
                            return send_msg(fd, eh, e.data(), e.size(), nullptr, 0, reply_err);
                        }
                    }
                    return send_msg(fd, r, src, back * 4, nullptr, 0, reply_err);
                });
                out_buf ^= 1;
                stats.bytes_out += sizeof r + back * 4;
                ++stats.prefills;
                stats.ms_prefill += work_ms;
                break;
            }
            case StageMsg::Reset: {
                std::string e;
                // an earlier prompt's commit (no reply of that request reported it); not this request's table
                if (!pending_err.empty()) {
                    std::fprintf(stderr, "strata stage worker: the last request's commit failed: %s\n",
                                 pending_err.c_str());
                    std::fflush(stderr);
                }
                pending_err.clear();
                // a chunk's reply still going out finishes first (a relay's reply thread reads the next worker's
                // link, which the reset may make again)
                if (!replies_out()) { keep = false; break; }
                if (!pos_err.empty()) {
                    e = pos_err;
                    pos_err.clear();
                    keep = reply_error(e);
                    break;
                }
                if (!h.reset(in.keep, e)) { keep = reply_error(e); break; }
                pos = 0;
                StageHeader r;
                r.type = (uint32_t) StageMsg::ResetOk;
                if (!send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                ++stats.resets;
                break;
            }
            case StageMsg::Positions: {
                // one way, like a commit: a failure is the next reply's error.  A chunk's reply still going out
                // finishes first (a relay forwards the table on the next worker's link, which it may make again).
                // The table is applied even with a Commit's error pending (a reset drops that), and its own failure
                // outlives a reset: the request it belongs to must not run on the old positions.  It replaces an
                // earlier table's failure (that request never started).
                std::string e;
                if (!replies_out()) { keep = false; break; }
                bool ok = false;
                if (!h.positions) e = "this worker takes no image positions (start it with --vision)";
                else ok = h.positions(in.h.b == 1 ? nullptr : in.tok32.data(), in.h.a, e);
                pos_err = ok ? std::string() : e.empty() ? std::string("the image positions failed") : e;
                break;
            }
            case StageMsg::CkptSave: {
                // after every chunk's reply and every window before it; a Commit that failed means the session is
                // not where the main process thinks, and so does another position
                std::string e;
                int stored = -1;
                if (!replies_out()) { keep = false; break; }
                if (!pending_err.empty() || !pos_err.empty()) e = take_pending();
                else if (in.h.b != pos)
                    e = "a checkpoint at " + std::to_string(in.h.b) + ", the worker's session is at " +
                        std::to_string(pos);
                else if (!h.ckpt_save) e = "this worker keeps no conversation checkpoints";
                else stored = h.ckpt_save(in.h.a, in.h.b, in.keep, e);
                if (stored < 0) { keep = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::CkptSaveOk;
                r.a = stored;
                if (!send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                break;
            }
            case StageMsg::CkptRestore: {
                std::string e;
                // the session is replaced: a Commit that failed before no longer matters (logged); a failed position
                // table does (it is this request's), as at a Reset
                if (!pending_err.empty()) {
                    std::fprintf(stderr, "strata stage worker: a commit failed before a checkpoint restore: %s\n",
                                 pending_err.c_str());
                    pending_err.clear();
                }
                if (!replies_out()) { keep = false; break; }
                if (!pos_err.empty()) {
                    e = pos_err;
                    pos_err.clear();
                    keep = reply_error(e);
                    break;
                }
                if (!h.ckpt_restore) e = "this worker keeps no conversation checkpoints";
                if (!h.ckpt_restore || !h.ckpt_restore(in.h.a, in.h.b, in.keep, e)) { keep = reply_error(e); break; }
                pos = in.h.b;
                StageHeader r;
                r.type = (uint32_t) StageMsg::CkptRestoreOk;
                if (!send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                break;
            }
            default:
                keep = reply_error("unknown message type " + std::to_string(in.h.type));
                break;
            }
            release();
            if (!keep) break;
        }
        (void) replies_out();   // a reply still going out ends first (or fails on the closed socket)
        // end the receiving thread: quit ends its polls and its waits on cv (it is set under mu, so a wait that checked
        // it just before cannot miss the notify), and shutting the socket down wakes a blocked recv on Linux.  On
        // Windows it does not: there only the rest of a message from an admitted peer (a header's rest, a body, or
        // the bytes a malformed one claims, drained) can still hold a recv, until that peer sends it or closes (or
        // its keepalive fails).
        {
            std::lock_guard<std::mutex> lk(mu);
            quit.store(true);
        }
        (void) sock_shutdown(fd);
        cv.notify_all();
        reader.join();
        (void) sock_close(fd);
        std::fprintf(stderr, "strata stage worker: %s disconnected%s%s\n", host.c_str(), err.empty() ? "" : ": ",
                     err.c_str());
        std::fflush(stderr);
        if (h.disconnected) h.disconnected();
        // a connection accepted meanwhile that did not take over (this one ended before its hello was decided): its
        // hello is served next, as a first one is
        if (cand != kNoSock) {
            Inbox ci = cand_hello.get();   // (within its 10 s)
            if (!ci.closed && ci.bad.empty() && (stop == nullptr || !*stop)) {
                next_fd = cand;
                next_host = cand_host;
                first = std::move(ci);
            } else {
                if (!ci.bad.empty()) send_error(cand, ci.bad);
                (void) sock_close(cand);
            }
            cand = kNoSock;
        }
    }
    if (next_fd != kNoSock) (void) sock_close(next_fd);   // (stopped with one waiting)
    (void) sock_close(lfd);
    return 0;
}

}  // namespace strata::net
