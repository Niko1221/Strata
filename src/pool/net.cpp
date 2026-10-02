// src/pool/net.cpp - the pool's sockets (Winsock / BSD) and the handshake's SHA-256 + HMAC.
#include "strata/pool/net.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t = SOCKET;
static inline int last_net_error() { return WSAGetLastError(); }
static inline void close_fd(intptr_t fd) { closesocket((SOCKET) fd); }
#define STRATA_SEND_FLAGS 0
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
using sock_t = int;
static inline int last_net_error() { return errno; }
static inline void close_fd(intptr_t fd) { ::close((int) fd); }
#define STRATA_SEND_FLAGS MSG_NOSIGNAL
#endif

namespace strata::pool {

namespace {

std::string error_text(int code) {
#if defined(_WIN32)
    char buf[256] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, (DWORD) code, 0, buf,
                   sizeof buf, nullptr);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s + " (" + std::to_string(code) + ")";
#else
    return std::string(std::strerror(code)) + " (" + std::to_string(code) + ")";
#endif
}

bool timed_out(int code) {
#if defined(_WIN32)
    return code == WSAETIMEDOUT;
#else
    return code == EAGAIN || code == EWOULDBLOCK || code == ETIMEDOUT;
#endif
}

void tune(sock_t fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (const char*) &one, sizeof one);
    int buf = 8 << 20;   // a prompt chunk is hundreds of MB: big buffers keep a fast LAN busy
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
}

std::string addr_text(const sockaddr* sa) {
    char host[INET6_ADDRSTRLEN] = {};
    int port = 0;
    if (sa->sa_family == AF_INET) {
        const sockaddr_in* a = (const sockaddr_in*) sa;
        inet_ntop(AF_INET, (void*) &a->sin_addr, host, sizeof host);
        port = ntohs(a->sin_port);
        return std::string(host) + ":" + std::to_string(port);
    }
    if (sa->sa_family == AF_INET6) {
        const sockaddr_in6* a = (const sockaddr_in6*) sa;
        inet_ntop(AF_INET6, (void*) &a->sin6_addr, host, sizeof host);
        port = ntohs(a->sin6_port);
        return "[" + std::string(host) + "]:" + std::to_string(port);
    }
    return "?";
}

void set_blocking(sock_t fd, bool blocking) {
#if defined(_WIN32)
    u_long mode = blocking ? 0 : 1;
    ioctlsocket(fd, FIONBIO, &mode);
#else
    const int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, blocking ? (fl & ~O_NONBLOCK) : (fl | O_NONBLOCK));
#endif
}

// wait until `fd` is readable (read) or writable; false on timeout or error (`err` set on error)
bool wait_fd(sock_t fd, bool read, double timeout_s, std::string& err) {
#if defined(_WIN32)
    WSAPOLLFD p{};
    p.fd = fd;
    p.events = read ? POLLRDNORM : POLLWRNORM;
    const int r = WSAPoll(&p, 1, timeout_s < 0 ? -1 : (int) (timeout_s * 1000.0));
#else
    pollfd p{};
    p.fd = fd;
    p.events = read ? POLLIN : POLLOUT;
    int r;
    do {
        r = ::poll(&p, 1, timeout_s < 0 ? -1 : (int) (timeout_s * 1000.0));
    } while (r < 0 && errno == EINTR);
#endif
    if (r < 0) { err = "poll: " + error_text(last_net_error()); return false; }
    return r > 0;
}

}  // namespace

bool net_init() {
#if defined(_WIN32)
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        WSADATA w;
        ok = WSAStartup(MAKEWORD(2, 2), &w) == 0;
    });
    return ok;
#else
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ Socket
Socket::~Socket() { close(); }

Socket& Socket::operator=(Socket&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_;
        peer_ = std::move(o.peer_);
        o.fd_ = -1;
    }
    return *this;
}

void Socket::close() {
    if (fd_ >= 0) {
#if defined(_WIN32)
        shutdown((SOCKET) fd_, SD_BOTH);
#else
        shutdown((int) fd_, SHUT_RDWR);
#endif
        close_fd(fd_);
        fd_ = -1;
    }
}

void Socket::set_timeout(double seconds) {
    if (fd_ < 0) return;
#if defined(_WIN32)
    DWORD ms = seconds > 0 ? (DWORD) (seconds * 1000.0) : 0;
    setsockopt((SOCKET) fd_, SOL_SOCKET, SO_RCVTIMEO, (const char*) &ms, sizeof ms);
    setsockopt((SOCKET) fd_, SOL_SOCKET, SO_SNDTIMEO, (const char*) &ms, sizeof ms);
#else
    timeval tv{};
    if (seconds > 0) {
        tv.tv_sec = (time_t) seconds;
        tv.tv_usec = (suseconds_t) ((seconds - (double) tv.tv_sec) * 1e6);
    }
    setsockopt((int) fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt((int) fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

bool Socket::send_all(const void* p, size_t n, std::string& err) {
    if (fd_ < 0) { err = "not connected"; return false; }
    const char* c = (const char*) p;
    while (n > 0) {
        const int part = (int) std::min<size_t>(n, (size_t) 1 << 30);
#if defined(_WIN32)
        const int s = ::send((SOCKET) fd_, c, part, STRATA_SEND_FLAGS);
#else
        const ssize_t s = ::send((int) fd_, c, (size_t) part, STRATA_SEND_FLAGS);
        if (s < 0 && errno == EINTR) continue;
#endif
        if (s <= 0) {
            const int e = last_net_error();
            err = timed_out(e) ? "send to " + peer_ + " timed out" : "send to " + peer_ + ": " + error_text(e);
            return false;
        }
        c += s;
        n -= (size_t) s;
    }
    return true;
}

bool Socket::recv_all(void* p, size_t n, std::string& err) {
    if (fd_ < 0) { err = "not connected"; return false; }
    char* c = (char*) p;
    while (n > 0) {
        const int part = (int) std::min<size_t>(n, (size_t) 1 << 30);
#if defined(_WIN32)
        const int r = ::recv((SOCKET) fd_, c, part, 0);
#else
        const ssize_t r = ::recv((int) fd_, c, (size_t) part, 0);
        if (r < 0 && errno == EINTR) continue;
#endif
        if (r == 0) { err = peer_ + " closed the connection"; return false; }
        if (r < 0) {
            const int e = last_net_error();
            err = timed_out(e) ? "no answer from " + peer_ + " (timed out)" : "receive from " + peer_ + ": " + error_text(e);
            return false;
        }
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ Listener
Listener::~Listener() { close(); }

void Listener::close() {
    if (fd_ >= 0) {
        close_fd(fd_);
        fd_ = -1;
    }
}

bool Listener::open(const std::string& host, int port, std::string& err) {
    if (!net_init()) { err = "the socket library did not start"; return false; }
    close();
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    const std::string h = host.empty() ? "0.0.0.0" : host;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(h.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + h;
        return false;
    }
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        sock_t fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
#if defined(_WIN32)
        if (fd == INVALID_SOCKET) continue;
        BOOL excl = TRUE;   // no second listener on the same port (Windows' SO_REUSEADDR would allow one)
        setsockopt(fd, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*) &excl, sizeof excl);
#else
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
        if (::bind(fd, a->ai_addr, (int) a->ai_addrlen) == 0 && ::listen(fd, 4) == 0) {
            fd_ = (intptr_t) fd;
            sockaddr_storage ss{};
            socklen_t sl = sizeof ss;
            if (getsockname(fd, (sockaddr*) &ss, &sl) == 0)
                port_ = ss.ss_family == AF_INET ? ntohs(((sockaddr_in*) &ss)->sin_port)
                                                : ntohs(((sockaddr_in6*) &ss)->sin6_port);
            freeaddrinfo(res);
            return true;
        }
        err = "cannot listen on " + h + ":" + ps + ": " + error_text(last_net_error());
        close_fd((intptr_t) fd);
    }
    freeaddrinfo(res);
    if (err.empty()) err = "cannot listen on " + h + ":" + ps;
    return false;
}

Socket Listener::accept(double timeout_s, std::string& err) {
    err.clear();
    if (fd_ < 0) { err = "not listening"; return Socket(); }
    if (!wait_fd((sock_t) fd_, true, timeout_s, err)) return Socket();
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    sock_t c = ::accept((sock_t) fd_, (sockaddr*) &ss, &sl);
#if defined(_WIN32)
    if (c == INVALID_SOCKET) { err = "accept: " + error_text(last_net_error()); return Socket(); }
#else
    if (c < 0) { err = "accept: " + error_text(last_net_error()); return Socket(); }
#endif
    tune(c);
    Socket s((intptr_t) c);
    s.set_peer(addr_text((sockaddr*) &ss));
    return s;
}

// ------------------------------------------------------------------------------------------------ connect
Socket connect_to(const std::string& host, int port, double timeout_s, std::string& err) {
    err.clear();
    if (!net_init()) { err = "the socket library did not start"; return Socket(); }
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return Socket();
    }
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        sock_t fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
#if defined(_WIN32)
        if (fd == INVALID_SOCKET) continue;
#else
        if (fd < 0) continue;
#endif
        set_blocking(fd, false);
        int r = ::connect(fd, a->ai_addr, (int) a->ai_addrlen);
        bool ok = r == 0;
        if (!ok) {
            const int e = last_net_error();
#if defined(_WIN32)
            const bool pending = e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
#else
            const bool pending = e == EINPROGRESS;
#endif
            if (pending) {
                std::string we;
                if (wait_fd(fd, false, timeout_s, we)) {
                    int so = 0;
                    socklen_t sl = sizeof so;
                    getsockopt(fd, SOL_SOCKET, SO_ERROR, (char*) &so, &sl);
                    ok = so == 0;
                    if (!ok) err = host + ":" + ps + ": " + error_text(so);
                } else {
                    err = we.empty() ? host + ":" + ps + ": no answer (timed out)" : we;
                }
            } else {
                err = host + ":" + ps + ": " + error_text(e);
            }
        }
        if (ok) {
            set_blocking(fd, true);
            tune(fd);
            Socket s((intptr_t) fd);
            s.set_peer(addr_text(a->ai_addr));
            freeaddrinfo(res);
            err.clear();
            return s;
        }
        close_fd((intptr_t) fd);
    }
    freeaddrinfo(res);
    if (err.empty()) err = "cannot connect to " + host + ":" + ps;
    return Socket();
}

bool split_host_port(const std::string& s0, int default_port, std::string& host, int& port) {
    std::string s = s0;
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    if (s.empty()) return false;
    port = default_port;
    if (s[0] == '[') {                              // [v6]:port
        const size_t e = s.find(']');
        if (e == std::string::npos) return false;
        host = s.substr(1, e - 1);
        if (e + 1 < s.size()) {
            if (s[e + 1] != ':') return false;
            const std::string p = s.substr(e + 2);
            if (p.empty() || p.find_first_not_of("0123456789") != std::string::npos) return false;
            port = std::atoi(p.c_str());
        }
    } else {
        const size_t c = s.rfind(':');
        if (c != std::string::npos && s.find(':') == c) {   // one colon: host:port
            host = s.substr(0, c);
            const std::string p = s.substr(c + 1);
            if (p.empty() || p.find_first_not_of("0123456789") != std::string::npos) return false;
            port = std::atoi(p.c_str());
        } else {
            host = s;                               // no colon, or a bare v6 address
        }
    }
    return !host.empty() && port > 0 && port < 65536;
}

// ------------------------------------------------------------------------------------------------ SHA-256
namespace {
constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64] = {};
    size_t used = 0;
    uint64_t total = 0;

    void block(const uint8_t* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void* data, size_t n) {
        const uint8_t* p = (const uint8_t*) data;
        total += n;
        while (n > 0) {
            const size_t take = std::min(n, 64 - used);
            std::memcpy(buf + used, p, take);
            used += take; p += take; n -= take;
            if (used == 64) { block(buf); used = 0; }
        }
    }
    std::array<uint8_t, 32> finish() {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (used != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(len, 8);
        std::array<uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) out[(size_t) (4 * i + j)] = (uint8_t) (h[i] >> (24 - 8 * j));
        return out;
    }
};
}  // namespace

std::array<uint8_t, 32> sha256(const void* data, size_t n) {
    Sha256 s;
    s.update(data, n);
    return s.finish();
}

std::array<uint8_t, 32> hmac_sha256(const std::string& key, const void* data, size_t n) {
    uint8_t k[64] = {};
    if (key.size() > 64) {
        const auto d = sha256(key.data(), key.size());
        std::memcpy(k, d.data(), d.size());
    } else {
        std::memcpy(k, key.data(), key.size());
    }
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) { ipad[i] = (uint8_t) (k[i] ^ 0x36); opad[i] = (uint8_t) (k[i] ^ 0x5c); }
    Sha256 in;
    in.update(ipad, 64);
    in.update(data, n);
    const auto inner = in.finish();
    Sha256 out;
    out.update(opad, 64);
    out.update(inner.data(), inner.size());
    return out.finish();
}

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; ++i) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}

std::string random_nonce() {
    std::random_device rd;
    uint8_t b[16];
    for (int i = 0; i < 16; i += 4) {
        const uint32_t v = rd() ^ (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count();
        std::memcpy(b + i, &v, 4);
    }
    return to_hex(b, sizeof b);
}

bool equal_ct(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char d = 0;
    for (size_t i = 0; i < a.size(); ++i) d |= (unsigned char) (a[i] ^ b[i]);
    return d == 0;
}

}  // namespace strata::pool
