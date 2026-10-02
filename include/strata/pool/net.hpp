// include/strata/pool/net.hpp - the pool: the TCP plumbing between pool nodes, and the SHA-256 / HMAC the
// handshake proves the shared secret with.
//
// A pool is one COORDINATOR (the PC that runs the server, the first layers, the head and the drafter) and one or
// more WORKERS (PCs that each run a contiguous range of the remaining layers).  Every worker listens on one TCP
// port; the coordinator connects to each of them.  Nothing here knows about the model: it moves bytes, with
// TCP_NODELAY (a verify window is a few hundred KB that must leave now, not after Nagle's 40 ms) and large socket
// buffers (a prompt chunk is hundreds of MB).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::pool {

/// A connected TCP stream.  Blocking, with a receive/send timeout; move-only.
class Socket {
public:
    Socket() = default;
    explicit Socket(intptr_t fd) : fd_(fd) {}
    ~Socket();
    Socket(Socket&& o) noexcept : fd_(o.fd_), peer_(std::move(o.peer_)) { o.fd_ = -1; }
    Socket& operator=(Socket&& o) noexcept;
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    bool valid() const { return fd_ >= 0; }
    void close();
    /// Sends all `n` bytes (false: the peer went away or the timeout passed; `err` says which).
    bool send_all(const void* p, size_t n, std::string& err);
    /// Receives exactly `n` bytes.
    bool recv_all(void* p, size_t n, std::string& err);
    /// Both directions' timeout in seconds (0 = none).
    void set_timeout(double seconds);
    /// "a.b.c.d:port" of the other end (set by connect/accept)
    const std::string& peer() const { return peer_; }
    void set_peer(std::string p) { peer_ = std::move(p); }

private:
    intptr_t fd_ = -1;
    std::string peer_;
};

/// A listening socket ("0.0.0.0", port).  `accept` waits up to `timeout_s` (< 0: forever); an invalid Socket
/// with `err` empty means the wait timed out.
class Listener {
public:
    Listener() = default;
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    bool open(const std::string& host, int port, std::string& err);
    Socket accept(double timeout_s, std::string& err);
    int port() const { return port_; }
    void close();

private:
    intptr_t fd_ = -1;
    int port_ = 0;
};

/// Connects to host:port within `timeout_s`.  An invalid Socket and `err` on failure.
Socket connect_to(const std::string& host, int port, double timeout_s, std::string& err);

/// "host:port" / "[v6]:port" / "host" (default port) -> parts.  false on a malformed address.
bool split_host_port(const std::string& s, int default_port, std::string& host, int& port);

/// One-time socket library start-up (WSAStartup on Windows; a no-op elsewhere).  Safe to call repeatedly.
bool net_init();

// ---- hashing for the handshake (no TLS: a pool runs on a LAN you trust; the secret keeps strangers' engines out)
std::array<uint8_t, 32> sha256(const void* data, size_t n);
std::array<uint8_t, 32> hmac_sha256(const std::string& key, const void* data, size_t n);
std::string to_hex(const uint8_t* p, size_t n);
/// 16 random bytes as hex (std::random_device; the handshake's nonce)
std::string random_nonce();
/// Constant-time comparison of two strings of equal length.
bool equal_ct(const std::string& a, const std::string& b);

}  // namespace strata::pool
