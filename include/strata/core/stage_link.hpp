#pragma once

// Two-PC Strata (2026-10): the stages of a layer split on ANOTHER PC.
//
// The DRIVER is the ordinary engine (embedding, the first layers, the head, the drafter, the server).  A range of
// layers [lb, le) in the middle of the model runs on a second PC, the STAGE: the same engine started with
// --stage-server, which loads only those layers (weights, experts, session) and has no head, no drafter and no
// server - it executes the windows the driver sends.  What crosses the cable per verify window is the hand-off
// buffer a layer split already passes between two cards (Verifier::set_stage), there and back.
//
// The STAGE dials the DRIVER (the driver listens): the driver's port is then the only one that has to be open, and
// the two engines can be started in either order.  One connection, blocking, TCP_NODELAY.  A failure anywhere ends
// the request with an error - there is no fallback that would change the numbers silently.

#include "strata/core/verify.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

enum class StageOp : uint32_t { Hello = 1, Run = 2, Commit = 3, WaitCommit = 4, Zero = 5, Ok = 100, Err = 101 };

/// What both sides must agree on before the first window.
struct StageHello {
    int32_t version = 1;
    int32_t max_t = 0;             ///< kVerifyMaxT: the rows of a hand-off buffer
    int64_t n_layers = 0, n_embd = 0, hc = 0, n_expert = 0;
    int64_t lb = 0, le = 0;        ///< the layers the stage runs
    uint64_t handoff_bytes = 0;    ///< one hand-off buffer
    uint64_t experts_total = 0;    ///< the pack's expert bytes: two different packs do not pass
};

/// One TCP connection.
class StageLink {
public:
    StageLink() = default;
    ~StageLink();
    StageLink(const StageLink&) = delete;
    StageLink& operator=(const StageLink&) = delete;

    /// the driver: listen on `port` (every interface; the firewall decides who may connect) and take ONE client,
    /// waiting up to `wait_s` seconds
    bool listen_accept(int port, int wait_s, std::string& err);
    /// the stage: connect to the driver, retrying for up to `wait_s` seconds
    bool connect_to(const std::string& host, int port, int wait_s, std::string& err);
    void close();
    bool is_open() const { return sock_ != -1; }

    /// header + up to two payload parts
    bool send_msg(StageOp op, uint32_t a, int64_t b, const void* p1, uint64_t n1, const void* p2, uint64_t n2,
                  std::string& err);
    bool recv_header(StageOp& op, uint32_t& a, int64_t& b, uint64_t& bytes, std::string& err);
    bool recv_bytes(void* p, uint64_t n, std::string& err);

private:
    intptr_t sock_ = -1;
};

/// The driver's side of the hop: after the stage before the gap has written its hand-off, send it, wait for the
/// remote layers, and put their hand-off where the stage after the gap reads it.
class RemoteStageBridge : public StageBridge {
public:
    /// listens, takes the stage's connection and checks its Hello against `mine` (lb / le are the gap)
    bool open(int port, int wait_s, const StageHello& mine, std::string& err);
    /// `src`: the hand-off the previous stage writes; `dst`: the one the next stage reads (both host memory)
    void set_buffers(const float* src, float* dst, uint64_t bytes) { src_ = src; dst_ = dst; bytes_ = bytes; }

    bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    bool wait_commit(std::string& err) override;
    /// a new sequence: the stage zeroes its session.  A failure is kept and returned by the next run().
    void zero();

    double ms_run = 0;      ///< wall time inside run(): the cable both ways + the remote layers
    int64_t windows = 0;

private:
    bool simple(StageOp op, uint32_t a, std::string& err);
    StageLink link_;
    const float* src_ = nullptr;
    float* dst_ = nullptr;
    uint64_t bytes_ = 0;
    std::string failed_;
};

}  // namespace strata::core
