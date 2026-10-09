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
// another thread.  POSIX sockets on Linux, Winsock on Windows.
#pragma once

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
inline constexpr uint32_t kStageProtocol = 3;

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
    char build[64] = {};          ///< engine build id (informational)
    char token[64] = {};          ///< the shared secret (STRATA_STAGE_TOKEN); the worker refuses a wrong one
};
#pragma pack(pop)

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
    bool connect(const std::string& host_port, const StageHello& mine, StageHello& peer, std::string& err);
    bool connected() const { return fd_ >= 0 && !broken_; }
    void close();

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

    const StageStats& stats() const { return stats_; }
    std::string peer_name() const { return peer_; }

private:
    bool read_reply_(StageMsg want, void* reply, size_t reply_bytes, int64_t& worker_us, std::string& err);
    /// a failed link: shut down (both directions fail from now on) - the descriptor is closed only by close(), so
    /// the other thread never reads or writes a number the process may have reused
    void break_();
    std::intptr_t fd_ = -1;   ///< the socket (a Winsock SOCKET fits; -1: none)
    bool broken_ = false;
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
    std::function<void()> disconnected;   ///< the main process went away (the session is stale)
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

/// Listens on `bind_addr`:`port` (empty: every interface) and serves one main process at a time until `stop` is set
/// or listening fails.  `token`: the secret a main process's hello must carry (empty: none - any host that reaches
/// the port can drive the worker).  Returns 0 when stopped, nonzero on a listen error (the reason on stderr).
int serve_stage(const std::string& bind_addr, int port, const std::string& token, const StageHandlers& h,
                const StageBuffers& buf, StageStats& stats, const bool* stop);

}  // namespace strata::net
