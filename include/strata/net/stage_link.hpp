// include/strata/net/stage_link.hpp - a layer split's later stage on another machine, over TCP.
//
// The main process runs layers [0, K) and the head; a worker process (strata --serve --stage-worker PORT
// --stage-begin K, on another PC) runs layers [K, n_layers) without the head.  They exchange exactly what an
// in-process layer split hands from one GPU to the next:
//   - a verify window: T rows of Verifier::handoff_floats (the residual streams, the pending write and the inject)
//     each way, then the accepted count (one way, no reply);
//   - a prompt chunk: T rows of hc * n_embd floats (the residual streams) each way.
// The worker keeps its own session (its layers' K/V, GDN state), expert cache and CPU expert pool; the main process
// sends a reset at every fresh prompt.  More than two stages: a relay worker (--stage-end K2 --stage-next HOST:PORT)
// runs [K, K2) and hands its rows to the next worker as the main process hands them to it; the next worker's reply
// is its reply (main -> A -> B -> A -> main), so the main process sees one worker either way.  Prompt chunks are pipelined: the main process sends chunk c + 1 while the
// worker reads chunk c (the worker receives on its own thread into two buffers), and takes the replies in order on
// another thread.  Images (--vision on both sides): a request with a picture rotates its tokens by M-RoPE positions
// that are not the cell index, so the main process sends its position table before the request (and a reset to the
// identity before the next text request); the worker's layers read their own copy.  POSIX sockets on Linux, Winsock on
// Windows.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace strata::net {

inline constexpr uint32_t kStageMagic = 0x4d525453u;   // "STRM"
inline constexpr uint32_t kStageProtocol = 5;

enum class StageMsg : uint32_t {
    Hello = 1,
    HelloOk = 2,
    Run = 3,        // a = T, b = pos0; payload: int32 tokens[T], float rows[T * handoff_floats]
    RunOk = 4,      // a = the worker's time (us); payload: float rows[T * handoff_floats]
    Commit = 5,     // a = n_keep; no reply (an error comes back on the next reply)
    Prefill = 6,    // a = T, b = pos0, c = flags (bit 0: the prompt is one chunk) | skip << 8; payload: int64 tokens[T],
                    // float rows[T * D]
    PrefillOk = 7,  // a = the worker's time (us); payload: float rows[(T - skip) * D], the chunk's rows from row `skip`
                    // (the main process needs no earlier ones: the drafter's window does not reach them)
    Reset = 8,      // a fresh prompt from position 0
    ResetOk = 9,
    Error = 10,     // payload: the message
    Positions = 11, // a = cells, b = 1: the identity (no payload); payload: int32 table[cells * 3], each cell's
                    // M-RoPE (t, h, w), for the next request; no reply (an error comes back on the next reply)
};

#pragma pack(push, 1)
struct StageHeader {
    uint32_t magic = kStageMagic;
    uint32_t type = 0;
    int64_t a = 0, b = 0, c = 0;
    uint64_t bytes = 0;   // payload bytes after the header
};

/// What both sides must agree on; exchanged once per connection, each side checks the other's.
struct StageHello {
    uint32_t protocol = kStageProtocol;
    int32_t n_embd = 0, hc = 0, n_layers = 0, n_expert = 0;
    int32_t layer_begin = 0;      ///< the worker's first layer (the main process's K)
    int32_t max_t = 0;            ///< the widest verify window
    int32_t kv_type = 0;          ///< the K/V element type id (both sides must store the same)
    int64_t max_context = 0;
    int64_t chunk = 0;            ///< the largest prompt chunk the main process sends
    int64_t handoff_floats = 0;   ///< floats per verify-window row
    uint64_t pack_hash = 0;       ///< the model pack's fingerprint (both sides must load the same model)
    int32_t tail_skip = 0;        ///< the main process's STRATA_ROUTE_TAIL_SKIP rank (0: off); a worker uses it
    int32_t vision = 0;           ///< started with --vision (a main process with it needs a worker with it)
    char build[56] = {};          ///< engine build id (informational)
    char token[64] = {};          ///< the shared secret (STRATA_STAGE_TOKEN); the worker refuses a wrong one
};
#pragma pack(pop)
// one size across protocols: an older peer's hello is read whole and refused for its protocol, not as malformed
static_assert(sizeof(StageHello) == 192, "the hello's size");

/// Seconds a send or receive on a link may block before the link counts as dead (STRATA_REMOTE_TIMEOUT_S, default 300).
int io_timeout_s();

/// Whether a StageClient's error says the link itself failed - nothing listened or the network did not reach it, the
/// connection closed, was reset or timed out, a send or a receive failed, the name did not resolve (yet: a name
/// service after a reboot) - so connecting again may mend it; false for a refusal (token, model, layers, chunk,
/// protocol: "refused", "differ"), an address that does not parse, and an error the worker reported
/// ("remote stage (worker): ...").
bool stage_link_lost(const std::string& err);

/// Whether a StageClient's error is the end of a wait its stop flag cancelled (StageClient::connect).
bool stage_wait_cancelled(const std::string& err);

/// Running totals of one side's traffic (STRATA_REMOTE_TIMING=1 prints them).
struct StageStats {
    uint64_t runs = 0, commits = 0, prefills = 0, resets = 0;
    uint64_t bytes_out = 0, bytes_in = 0;
    double ms_run = 0, ms_prefill = 0;   ///< wall time of the round trips (main side) / of the work (worker side)
    double ms_run_worker = 0;            ///< main side: the worker's own share of ms_run
};

/// The main process's end: one connection.  Windows, commits and resets are whole round trips (one thread at a
/// time); prompt chunks are a send (prefill_send) and, on another thread, a receive (prefill_recv) in the same order.
class StageClient {
public:
    StageClient() = default;
    ~StageClient();
    StageClient(const StageClient&) = delete;
    StageClient& operator=(const StageClient&) = delete;

    /// "host:port".  Sends `mine`, receives the worker's hello into `peer` and checks it against `mine`.
    /// `timeout_s` > 0: the connection and the hello's reply take that long at most, together; an address that does
    /// not answer leaves the next one of the name's addresses the rest (each but the last gets 10 s at most).  0: the
    /// system's connect time-out, then STRATA_REMOTE_TIMEOUT_S for the reply.  `stop`, set by another thread, ends the
    /// wait for either: "remote stage: the wait was cancelled" (stage_wait_cancelled), which stage_link_lost does not
    /// count.  The name's lookup is not cut short by either.
    bool connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err,
                 int timeout_s = 0, const std::atomic<bool>* stop = nullptr);
    bool connected() const { return fd_ >= 0 && !broken_; }
    void close();
    /// Between requests (no reply outstanding): whether the link still stands, without sending anything and without
    /// blocking.  The worker sends nothing unasked, so a readable socket means its end closed or failed, or bytes
    /// came that no message asked for (the link is out of step): false, with an error stage_link_lost counts, and
    /// the link is marked failed.
    bool alive(std::string& err);
    /// From another thread: the socket shut down, so a run() waiting for its reply fails (and then breaks the link).
    /// Linux wakes a blocked read at once; on Windows (not tested) the read may wait for the link's own timeout.
    void abort();

    /// One verify window: send T tokens and T rows (in_floats floats), receive T rows (out_floats floats).
    bool run(int T, const int32_t* tokens, int64_t pos0, const float* rows_in, size_t in_floats, float* rows_out,
             size_t out_floats, std::string& err);
    /// The accepted count of the last window (one way).
    bool commit(int n_keep, std::string& err);
    /// One prompt chunk out: T tokens and T rows of `row_floats`; its reply is the next prefill_recv's.
    bool prefill_send(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows_in,
                      size_t row_floats, int64_t skip, std::string& err);
    /// The oldest outstanding chunk's reply: rows [skip, T) into rows_out + skip rows.
    bool prefill_recv(float* rows_out, size_t row_floats, int64_t T, int64_t skip, std::string& err);
    /// A fresh prompt: the worker zeroes its session.
    bool reset(std::string& err);
    /// The positions of the next request's cells [0, cells): `table` is (t, h, w) per cell; nullptr: the identity
    /// (every cell its own index, as a text request has).  One way.
    bool positions(const int32_t* table, int64_t cells, std::string& err);

    const StageStats& stats() const { return stats_; }
    std::string peer_name() const { return peer_; }

private:
    bool read_reply_(StageMsg want, void* reply, size_t reply_bytes, int64_t& worker_us, std::string& err);
    /// a failed link: shut down (both directions fail from now on) - the descriptor is closed only by close(), so
    /// the other thread never reads or writes a number the process may have reused
    void break_();
    std::intptr_t fd_ = -1;   ///< the socket (a Winsock SOCKET fits; -1: none)
    std::atomic<bool> broken_{false};   ///< (the sending side sets it, the receiving side reads it)
    std::string peer_;
    std::mutex send_mu_, recv_mu_, q_mu_;
    std::deque<std::chrono::steady_clock::time_point> sent_at_;   ///< outstanding chunks' send start (timing)
    StageStats stats_;
};

/// A relay worker's forwarding of prompt chunks to the next worker: one send in flight, in order.  `rows` must stay
/// untouched until the next send() or wait() returns (the prompt path's two hand-off buffers alternate, so the one a
/// chunk writes was sent two chunks ago - and send() waited for that).
class StageRelay {
public:
    explicit StageRelay(StageClient& next) : next_(next) {}
    ~StageRelay();
    StageRelay(const StageRelay&) = delete;
    StageRelay& operator=(const StageRelay&) = delete;
    /// waits for the previous send, then sends this chunk on a thread
    bool send(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, const float* rows, size_t row_floats,
              int64_t skip, std::string& err);
    /// the send in flight is out (or failed: its error)
    bool wait(std::string& err);

private:
    StageClient& next_;
    std::future<bool> out_;
    std::string err_;
    std::vector<int64_t> tokens_;
};

/// The worker's end.  Every handler runs on the serving thread, one message at a time in arrival order.
struct StageHandlers {
    std::function<bool(const StageHello& main_hello, StageHello& mine, std::string& err)> hello;
    /// rows in: StageBuffers::run_in; rows out: run_out
    std::function<bool(int T, const int32_t* tokens, int64_t pos0, std::string& err)> run;
    std::function<bool(int n_keep, std::string& err)> commit;
    /// rows in: `rows_in` (one of StageBuffers::pf_in[2]); rows out: `rows_out` (one of pf_out[2], all T rows).
    /// `skip`: the rows the reply leaves out (the main process does not read them)
    std::function<bool(const int64_t* tokens, int64_t T, int64_t pos0, int64_t flags, int64_t skip, const float* rows_in,
                       float* rows_out, std::string& err)> prefill;
    /// optional (a relay worker): fills a chunk's rows_out [skip, T) just before its reply goes out, on the replying
    /// thread, in chunk order - the serving thread reads the next chunk meanwhile.  A failure sends an error instead.
    std::function<bool(float* rows_out, int64_t T, int64_t skip, std::string& err)> prefill_reply;
    std::function<bool(std::string& err)> reset;
    /// optional (--vision): the next request's positions, as StageClient::positions sends them (table nullptr: the
    /// identity).  Unset: a Positions message is an error.
    std::function<bool(const int32_t* table, int64_t cells, std::string& err)> positions;
    std::function<void()> disconnected;   ///< the main process went away (the session is stale)
    /// optional (a relay worker): about once a second while the served main process is idle (no message read or being
    /// read, no reply going out), on the serving thread.  False ends that connection with `err`, as a closed one ends:
    /// its main process finds the link lost at its next request and connects again.
    std::function<bool(std::string& err)> idle;
};

struct StageBuffers {
    float* run_in = nullptr;    ///< max_t * handoff_floats (device-visible: the verifier reads it)
    float* run_out = nullptr;
    size_t run_floats = 0;      ///< capacity of each, floats
    float* pf_in[2] = {};       ///< chunk * D each: the receiving thread fills one while the other is read
    float* pf_out[2] = {};      ///< chunk * D each: one chunk's rows go back while the next chunk's are written
    size_t pf_floats = 0;
    int64_t handoff_floats = 0; ///< floats per verify row
    int64_t pf_row_floats = 0;  ///< floats per prompt row (D)
};

/// A worker's position table (`cells` x (t, h, w)) from a Positions message: cells [0, cells_in) from `table`, the
/// rest continuing from the last one sent, as the main process's own table continues past its prompt; `table` nullptr:
/// the identity (every cell its own index).
inline void stage_positions_fill(int32_t* out, int64_t cells, const int32_t* table, int64_t cells_in) {
    for (int64_t c = 0; c < cells; ++c) {
        int32_t* p = out + c * 3;
        if (table == nullptr) {
            p[0] = p[1] = p[2] = (int32_t) c;
        } else if (c < cells_in) {
            p[0] = table[c * 3];
            p[1] = table[c * 3 + 1];
            p[2] = table[c * 3 + 2];
        } else {
            p[0] = p[1] = p[2] = (int32_t) ((int64_t) table[(cells_in - 1) * 3] + (c - cells_in + 1));
        }
    }
}

/// Listens on `bind_addr`:`port` (empty: every interface) and serves one main process at a time until `stop` is set
/// or listening fails.  A new connection waits in the listen backlog until the served one has had no message read,
/// being read, or a reply going out for STRATA_REMOTE_TAKEOVER_S seconds (default 30; 0: until it closes); then a
/// valid hello takes the worker over: the old one ends as a closed one does (`disconnected`).  An invalid one is
/// refused and the old one stays; one accepted while the served one sends again is closed (its main process connects
/// again).  `token`: the secret a main process's hello must carry (empty: none - any host that reaches the port can
/// drive the worker).  Returns 0 when stopped, nonzero on a listen error (the reason on stderr).
int serve_stage(const std::string& bind_addr, int port, const std::string& token, const StageHandlers& h,
                const StageBuffers& buf, StageStats& stats, const std::atomic<bool>* stop);

}  // namespace strata::net
