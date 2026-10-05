// Two-PC Strata: the socket between the driver engine and the stage engine (see stage_link.hpp).
// winsock2.h has to come before anything that pulls in windows.h, which is why this is its own file.

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
#define STRATA_CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
#define STRATA_CLOSESOCK ::close
#endif

#include "strata/core/stage_link.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace strata::core {

namespace {

constexpr uint32_t kMagic = 0x31475453;   // "STG1"

struct Header {
    uint32_t magic;
    uint32_t op;
    uint32_t a;
    uint32_t pad;
    int64_t b;
    uint64_t bytes;
};
static_assert(sizeof(Header) == 32, "the stage link header is 32 bytes on both PCs");

bool net_up(std::string& err) {
#if defined(_WIN32)
    static const int rc = [] { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }();
    if (rc != 0) { err = "stage link: WSAStartup failed"; return false; }
#endif
    (void) err;
    return true;
}

void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 4 << 20;   // a hand-off buffer is a few hundred KB: let one fit the socket buffers
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
}

bool send_all(sock_t s, const void* p, uint64_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int k = (int) ::send(s, c, (int) std::min<uint64_t>(n, 1u << 30), 0);
        if (k <= 0) return false;
        c += k;
        n -= (uint64_t) k;
    }
    return true;
}

bool recv_all(sock_t s, void* p, uint64_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const int k = (int) ::recv(s, c, (int) std::min<uint64_t>(n, 1u << 30), 0);
        if (k <= 0) return false;
        c += k;
        n -= (uint64_t) k;
    }
    return true;
}

}  // namespace

StageLink::~StageLink() { close(); }

void StageLink::close() {
    if (sock_ != -1) STRATA_CLOSESOCK((sock_t) sock_);
    sock_ = -1;
}

bool StageLink::listen_accept(int port, int wait_s, std::string& err) {
    close();
    if (!net_up(err)) return false;
    const sock_t ls = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls == (sock_t) -1) { err = "stage link: no socket"; return false; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t) port);
    if (::bind(ls, (const sockaddr*) &a, sizeof a) != 0 || ::listen(ls, 1) != 0) {
        STRATA_CLOSESOCK(ls);
        err = "stage link: cannot listen on port " + std::to_string(port) + " (in use?)";
        return false;
    }
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(ls, &fds);
    timeval tv{};
    tv.tv_sec = wait_s;
    const int r = ::select((int) ls + 1, &fds, nullptr, nullptr, &tv);
    if (r <= 0) {
        STRATA_CLOSESOCK(ls);
        err = "stage link: no stage connected to port " + std::to_string(port) + " within " + std::to_string(wait_s) + " s";
        return false;
    }
    const sock_t s = ::accept(ls, nullptr, nullptr);
    STRATA_CLOSESOCK(ls);
    if (s == (sock_t) -1) { err = "stage link: accept failed"; return false; }
    tune(s);
    sock_ = (intptr_t) s;
    return true;
}

bool StageLink::connect_to(const std::string& host, int port, int wait_s, std::string& err) {
    close();
    if (!net_up(err)) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) {
        err = "stage link: " + host + " is not an IPv4 address";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == (sock_t) -1) { err = "stage link: no socket"; return false; }
        if (::connect(s, (const sockaddr*) &a, sizeof a) == 0) {
            tune(s);
            sock_ = (intptr_t) s;
            return true;
        }
        STRATA_CLOSESOCK(s);
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > (double) wait_s) {
            err = "stage link: no driver at " + host + ":" + std::to_string(port) + " within " + std::to_string(wait_s) + " s";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

bool StageLink::send_msg(StageOp op, uint32_t a, int64_t b, const void* p1, uint64_t n1, const void* p2, uint64_t n2,
                         std::string& err) {
    if (sock_ == -1) { err = "stage link: not connected"; return false; }
    Header h{kMagic, (uint32_t) op, a, 0, b, n1 + n2};
    if (!send_all((sock_t) sock_, &h, sizeof h) || (n1 > 0 && !send_all((sock_t) sock_, p1, n1)) ||
        (n2 > 0 && !send_all((sock_t) sock_, p2, n2))) {
        err = "stage link: the connection to the other PC was lost (send)";
        close();
        return false;
    }
    return true;
}

bool StageLink::recv_header(StageOp& op, uint32_t& a, int64_t& b, uint64_t& bytes, std::string& err) {
    if (sock_ == -1) { err = "stage link: not connected"; return false; }
    Header h{};
    if (!recv_all((sock_t) sock_, &h, sizeof h)) {
        err = "stage link: the connection to the other PC was lost (receive)";
        close();
        return false;
    }
    if (h.magic != kMagic) { err = "stage link: not a Strata stage on the other end"; close(); return false; }
    op = (StageOp) h.op;
    a = h.a;
    b = h.b;
    bytes = h.bytes;
    return true;
}

bool StageLink::recv_bytes(void* p, uint64_t n, std::string& err) {
    if (sock_ == -1) { err = "stage link: not connected"; return false; }
    if (n > 0 && !recv_all((sock_t) sock_, p, n)) {
        err = "stage link: the connection to the other PC was lost (payload)";
        close();
        return false;
    }
    return true;
}

// ---- the driver's bridge

namespace {
/// the reply to a request: Ok with `want` payload bytes into `into`, or the stage's error text
bool take_reply(StageLink& link, void* into, uint64_t want, std::string& err) {
    StageOp op{};
    uint32_t a = 0;
    int64_t b = 0;
    uint64_t bytes = 0;
    if (!link.recv_header(op, a, b, bytes, err)) return false;
    if (op == StageOp::Err) {
        std::string text((size_t) std::min<uint64_t>(bytes, 4096), '\0');
        if (!link.recv_bytes(text.data(), text.size(), err)) return false;
        err = "the stage on the other PC: " + text;
        return false;
    }
    if (op != StageOp::Ok || bytes != want) {
        err = "stage link: an unexpected reply from the other PC";
        link.close();
        return false;
    }
    return link.recv_bytes(into, want, err);
}
}  // namespace

bool RemoteStageBridge::open(int port, int wait_s, const StageHello& mine, std::string& err) {
    // Anything may knock on an open port (a health check of whatever used this port before): a caller that does not
    // answer the hello as a stage is dropped and the wait goes on, until the stage itself arrives or the time is up.
    StageHello theirs{};
    const auto t0 = std::chrono::steady_clock::now();
    for (int strangers = 0;; ++strangers) {
        const int left = wait_s - (int) std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (left <= 0) {
            err = "stage link: no stage connected to port " + std::to_string(port) + " within " + std::to_string(wait_s) +
                  " s (" + std::to_string(strangers) + " other callers were turned away)";
            return false;
        }
        if (!link_.listen_accept(port, left, err)) return false;
        if (link_.send_msg(StageOp::Hello, 0, 0, &mine, sizeof mine, nullptr, 0, err) &&
            take_reply(link_, &theirs, sizeof theirs, err))
            break;
        link_.close();
    }
    auto bad = [&](const char* what, long long m, long long t) {
        err = std::string("the stage on the other PC does not match this engine: ") + what + " " + std::to_string(m) +
              " here, " + std::to_string(t) + " there";
        link_.close();
        return false;
    };
    if (theirs.version != mine.version) return bad("protocol version", mine.version, theirs.version);
    if (theirs.max_t != mine.max_t) return bad("window rows", mine.max_t, theirs.max_t);
    if (theirs.n_layers != mine.n_layers) return bad("layers", mine.n_layers, theirs.n_layers);
    if (theirs.n_embd != mine.n_embd) return bad("width", mine.n_embd, theirs.n_embd);
    if (theirs.hc != mine.hc) return bad("hyper-connection streams", mine.hc, theirs.hc);
    if (theirs.n_expert != mine.n_expert) return bad("experts per layer", mine.n_expert, theirs.n_expert);
    if (theirs.handoff_bytes != mine.handoff_bytes)
        return bad("hand-off bytes", (long long) mine.handoff_bytes, (long long) theirs.handoff_bytes);
    if (theirs.experts_total != mine.experts_total)
        return bad("expert bytes of the pack", (long long) mine.experts_total, (long long) theirs.experts_total);
    if (theirs.lb != mine.lb) return bad("first remote layer", mine.lb, theirs.lb);
    if (theirs.le != mine.le) return bad("end of the remote layers", mine.le, theirs.le);
    return true;
}

bool RemoteStageBridge::run(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    if (!failed_.empty()) { err = failed_; return false; }
    if (src_ == nullptr || dst_ == nullptr) { err = "stage link: no hand-off buffers"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    int32_t tok[16] = {};
    for (int t = 0; t < T && t < 16; ++t) tok[t] = tokens[t];
    if (!link_.send_msg(StageOp::Run, (uint32_t) T, pos0, tok, sizeof tok, src_, bytes_, err) ||
        !take_reply(link_, dst_, bytes_, err)) {
        failed_ = err;
        return false;
    }
    ms_run += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++windows;
    return true;
}

bool RemoteStageBridge::simple(StageOp op, uint32_t a, std::string& err) {
    if (!failed_.empty()) { err = failed_; return false; }
    if (!link_.send_msg(op, a, 0, nullptr, 0, nullptr, 0, err) || !take_reply(link_, nullptr, 0, err)) {
        failed_ = err;
        return false;
    }
    return true;
}

bool RemoteStageBridge::commit(int n_keep, std::string& err) { return simple(StageOp::Commit, (uint32_t) n_keep, err); }
bool RemoteStageBridge::wait_commit(std::string& err) { return simple(StageOp::WaitCommit, 0, err); }
void RemoteStageBridge::zero() {
    std::string e;
    (void) simple(StageOp::Zero, 0, e);
}

}  // namespace strata::core
