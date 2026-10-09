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
#endif

namespace strata::net {

namespace {

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
constexpr int kSendFlags = MSG_NOSIGNAL;   // a closed peer is an error, not SIGPIPE
constexpr int kListenOpt = SO_REUSEADDR;   // a restart binds over its own TIME_WAIT, never over a live listener
int sock_close(sock_t s) { return ::close(s); }
int sock_shutdown(sock_t s) { return ::shutdown(s, SHUT_RDWR); }
int sock_errno() { return errno; }
bool sock_eintr(int e) { return e == EINTR; }
bool sock_timedout(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
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

void tune_socket(sock_t fd) {
    set_opt<int>(fd, IPPROTO_TCP, TCP_NODELAY, 1);
    set_opt<int>(fd, SOL_SOCKET, SO_KEEPALIVE, 1);
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

}  // namespace

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

bool StageClient::connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err) {
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
    for (addrinfo* ai = res; ai != nullptr && fd == kNoSock; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == kNoSock) { last = sock_errno(); continue; }
        if (::connect(fd, ai->ai_addr, (sock_len) ai->ai_addrlen) != 0) {
            last = sock_errno();
            (void) sock_close(fd);
            fd = kNoSock;
        }
    }
    freeaddrinfo(res);
    if (fd == kNoSock) { err = "remote stage: cannot connect to " + host_port + ": " + sock_errstr(last); return false; }
    tune_socket(fd);
    fd_ = (std::intptr_t) fd;
    peer_ = host_port;
    StageHeader h;
    h.type = (uint32_t) StageMsg::Hello;
    h.bytes = sizeof mine;
    StageHeader r;
    if (!send_msg(S(fd_), h, &mine, sizeof mine, nullptr, 0, err) || !recv_header(S(fd_), r, err)) { close(); return false; }
    if (r.type == (uint32_t) StageMsg::Error) {
        std::string m((size_t) r.bytes, '\0');
        std::string e2;
        (void) recv_all(S(fd_), m.data(), m.size(), e2);
        err = "remote stage: the worker refused: " + m;
        close();
        return false;
    }
    if (r.type != (uint32_t) StageMsg::HelloOk || r.bytes != sizeof peer || !recv_all(S(fd_), &peer, sizeof peer, err)) {
        if (err.empty()) err = "remote stage: unexpected reply to hello";
        close();
        return false;
    }
    if (!hello_matches(mine, peer, err)) { close(); return false; }
    return true;
}

// the next reply: `want` with exactly reply_bytes of payload, or the worker's error (the connection then stays)
bool StageClient::read_reply_(StageMsg want, void* reply, size_t reply_bytes, int64_t& worker_us, std::string& err) {
    StageHeader r;
    if (broken_) { err = "remote stage: the link failed earlier"; return false; }
    if (!recv_header(S(fd_), r, err)) { break_(); return false; }
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

bool StageClient::reset(std::string& err) {
    std::lock_guard<std::mutex> ls(send_mu_);
    std::lock_guard<std::mutex> lr(recv_mu_);
    if (!connected()) { err = "remote stage: not connected"; return false; }
    StageHeader h;
    h.type = (uint32_t) StageMsg::Reset;
    if (!send_msg(S(fd_), h, nullptr, 0, nullptr, 0, err)) { break_(); return false; }
    int64_t wus = 0;
    ++stats_.resets;
    return read_reply_(StageMsg::ResetOk, nullptr, 0, wus, err);
}

// ------------------------------------------------------------------------------------------------ serve_stage

namespace {

// One message as the receiving thread read it.  A prompt chunk's rows are already in buf.pf_in[slot].
struct Inbox {
    StageHeader h;
    std::vector<int32_t> tok32;
    std::vector<int64_t> tok64;
    std::vector<float> rows;    ///< a window's rows (copied to run_in by the serving thread)
    StageHello hello;
    int slot = -1;
    std::string bad;            ///< the message was malformed: reply this error
    bool closed = false;        ///< the connection ended (err says why)
    std::string err;
};

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
                const StageBuffers& buf, StageStats& stats, const bool* stop) {
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
    while (stop == nullptr || !*stop) {
        poll_fd pfd{};
        pfd.fd = lfd;
        pfd.events = POLLIN;
        const int pr = sock_poll(&pfd, 1, 1000);
        if (pr < 0 && !sock_eintr(sock_errno())) {   // (a network that went down: no spinning on the error)
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        if (pr <= 0) continue;
        sockaddr_storage peer{};
        sock_len plen = sizeof peer;
        const sock_t fd = ::accept(lfd, (sockaddr*) &peer, &plen);
        if (fd == kNoSock) continue;
        char host[128] = "?";
        (void) getnameinfo((sockaddr*) &peer, plen, host, sizeof host, nullptr, 0, NI_NUMERICHOST);
        tune_socket(fd);
        // the hello within 10 s (a silent peer does not hold the only connection); after it no read timeout - the
        // main process may wait for its next prompt for hours (the receiving thread lifts it once the hello is in)
        set_timeout(fd, SO_RCVTIMEO, 10);
        std::fprintf(stderr, "strata stage worker: connection from %s\n", host);
        std::fflush(stderr);

        // ---- the receiving thread: whole messages into a queue, a prompt chunk's rows straight into a free pf_in
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Inbox> q;
        bool slot_busy[2] = {false, false};
        std::atomic<bool> quit{false};
        std::thread reader([&] {
            for (;;) {
                Inbox in;
                if (!recv_header(fd, in.h, in.err)) {
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
                        const int64_t T = in.h.a;
                        const size_t rows = (size_t) std::max<int64_t>(T, 0) * (size_t) buf.handoff_floats;
                        if (T < 1 || rows > buf.run_floats || in.h.bytes != (uint64_t) T * 4 + rows * 4) {
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
                        const int64_t T = in.h.a;
                        const size_t rows = (size_t) std::max<int64_t>(T, 0) * (size_t) buf.pf_row_floats;
                        if (T < 1 || rows > buf.pf_floats || in.h.bytes != (uint64_t) T * 8 + rows * 4) {
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
                    } else if (type != StageMsg::Commit && type != StageMsg::Reset) {
                        in.bad = "unknown message type " + std::to_string(in.h.type);
                        if (!drain(fd, in.h.bytes, in.err)) in.closed = true;
                    }
                }
                const bool closed = in.closed;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(in));
                }
                cv.notify_all();
                if (closed || quit.load()) break;
            }
        });

        // ---- the serving thread (this one): the messages in order
        std::string err, pending_err;   // pending_err: a failed one-way message, reported on the next reply
        bool greeted = false;
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
        for (;;) {
            Inbox in;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty(); });
                in = std::move(q.front());
                q.pop_front();
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
            stats.bytes_in += sizeof in.h + in.h.bytes;
            if (!greeted && type != StageMsg::Hello) { (void) reply_error("hello first"); release(); break; }
            if (!in.bad.empty()) {
                release();
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
                    std::fprintf(stderr, "strata stage worker: %s sent a wrong token: refused\n", host);
                    (void) reply_error("wrong token (STRATA_STAGE_TOKEN)");
                    keep = false;
                    break;
                }
                if (!h.hello(in.hello, mine, he) || !hello_matches(in.hello, mine, he)) {
                    (void) reply_error(he);
                    keep = false;
                    break;
                }
                StageHeader r;
                r.type = (uint32_t) StageMsg::HelloOk;
                r.bytes = sizeof mine;
                if (!replies_out() || !send_msg(fd, r, &mine, sizeof mine, nullptr, 0, err)) keep = false;
                greeted = true;
                std::fprintf(stderr, "strata stage worker: main process %s: layers %d.. here, context %lld, chunk %lld\n",
                             host, in.hello.layer_begin, (long long) in.hello.max_context, (long long) in.hello.chunk);
                std::fflush(stderr);
                break;
            }
            case StageMsg::Run: {
                const int T = (int) in.h.a;
                std::memcpy(buf.run_in, in.rows.data(), in.rows.size() * 4);
                std::string e;
                bool ok;
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.run(T, in.tok32.data(), in.h.b, e);
                if (!ok) { keep = reply_error(e); break; }
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
                if (!pending_err.empty()) { e = pending_err; pending_err.clear(); ok = false; }
                else ok = h.prefill(in.tok64.data(), T, in.h.b, in.h.c & 0xff, skip, buf.pf_in[in.slot], out, e);
                release();   // the chunk is read (its rows were uploaded before `prefill` returned)
                if (!ok) { keep = reply_error(e); break; }
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
                pending_err.clear();
                // a chunk's reply still going out finishes first (a relay's reply thread reads the next worker's
                // link, which the reset may make again)
                if (!replies_out()) { keep = false; break; }
                if (!h.reset(e)) { keep = reply_error(e); break; }
                StageHeader r;
                r.type = (uint32_t) StageMsg::ResetOk;
                if (!send_msg(fd, r, nullptr, 0, nullptr, 0, err)) keep = false;
                ++stats.resets;
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
        // end the receiving thread: shutting the socket down wakes a blocked recv
        quit.store(true);
        (void) sock_shutdown(fd);
        cv.notify_all();
        reader.join();
        (void) sock_close(fd);
        std::fprintf(stderr, "strata stage worker: %s disconnected%s%s\n", host, err.empty() ? "" : ": ", err.c_str());
        std::fflush(stderr);
        if (h.disconnected) h.disconnected();
    }
    (void) sock_close(lfd);
    return 0;
}

}  // namespace strata::net
