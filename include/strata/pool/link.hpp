// include/strata/pool/link.hpp - the pool's coordinator side: the workers, and the head that runs here.
//
// The coordinator's own verifier and prompt path run layers [0, K1) and stop there (`Verifier::set_link`,
// `Prefill::remote_next`).  PoolLink carries their hand-off to each worker in turn (a star: coordinator -> worker 1
// -> coordinator -> worker 2 ...; with the usual two PCs that is one hop out and one back) and brings the final
// residual home, where it runs the output head and the sampler (the same kernels the verify window's head records)
// and keeps the final rows where the draft layer (MTP) reads them.  The drafter, the sampler and every decision of
// the serve loop therefore stay on the coordinator, exactly as with one GPU; a worker is a pure function of the
// rows it gets and of its own layers' state.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/verify.hpp"
#include "strata/pool/protocol.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core { class NativeHead; }

namespace strata::pool {

struct WorkerInfo {
    std::string addr;        ///< as configured (host:port)
    KV hello;                ///< what it said about itself
    int64_t lb = 0, le = 0;  ///< its layers once configured
    KV ready;                ///< READY: slots, cache_mib, chunk, vram_free_mib, arena_mib
    KV req;                  ///< its own numbers for the last request (the END_REQUEST ack): windows, ms_verify, ...
    double ms_rtt = 0;       ///< that ack's round trip (the network's latency, near enough)
};

/// What the coordinator knows about the model, to compare with each worker's HELLO.
struct ModelFingerprint {
    int64_t n_layers = 0, n_expert = 0, k = 0, n_embd = 0, hc = 0;
    uint64_t experts_bytes = 0, dense_bytes = 0;
    int native = 0;
    std::string pack;        ///< the pack folder's name (a hint in messages only)
    std::string engine;      ///< engine version
    KV to_kv() const;
    /// "" when `w` describes the same model, else what differs
    std::string mismatch(const KV& w) const;
};

class PoolLink final : public core::StageLink {
public:
    PoolLink() = default;
    ~PoolLink() override;
    PoolLink(const PoolLink&) = delete;
    PoolLink& operator=(const PoolLink&) = delete;

    /// Connects to every worker (retrying for up to `wait_s` while they start), reads its HELLO and proves the shared
    /// secret both ways.  `timeout_s` bounds every later reply.
    bool connect(const std::vector<std::string>& peers, const std::string& secret, const ModelFingerprint& fp,
                 double wait_s, double timeout_s, std::string& err);
    std::vector<WorkerInfo>& workers() { return workers_; }
    size_t size() const { return workers_.size(); }

    /// CONFIG to every worker: worker i runs [at[i], at[i + 1]) (the last to n_layers).  `common` carries the
    /// settings every worker must share (context, KV format, window, rope, wire).
    bool configure(const std::vector<int64_t>& at, int64_t n_layers, const KV& common, Wire wire, Wire draft_wire,
                   std::string& err);
    /// Waits for every READY (each worker loads its share meanwhile; minutes for a big range).
    bool wait_ready(double timeout_s, std::string& err);
    /// the smallest prompt chunk any worker can take (0 when none said)
    int64_t max_chunk() const;

    /// The device side, on the coordinator's GPU: the buffers the head writes into, the window's hand-off (host
    /// pointer of the mapped buffer the coordinator's verifier writes), and the prompt rows.  After `wait_ready`.
    bool init_device(const core::WeightTable& wt, const core::ModelGeometry& g, const core::BlockBuffers& block,
                     const core::NativeHead* head, int max_t, int64_t n_vocab, int64_t max_chunk,
                     const float* handoff_host, std::string& err);

    // ---- core::StageLink: the rest of a verify window
    bool run(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    const float* final_R_all() const override { return R_dev_; }
    void set_sampling(const strata::kernels::SamplerParams& sp) override { sampling_ = sp; }
    void set_history(const int32_t* history, int history_len) override { hist_d_ = history; hist_len_ = history_len; }
    void set_head_sampling(bool on) override { head_sampling_ = on; }

    /// The rest of a prompt chunk: `rows` (host, T x hc*n_embd float32) through every worker; `*out` then points at
    /// the final rows (host, float32, valid until the next call) for the draft layer.  `segment` is the length of the
    /// batched part this chunk belongs to (a worker lends its cache slots for that); `ckpt` != 0 asks every worker to
    /// keep its state after this chunk under that id.
    bool prefill(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, int64_t segment, uint64_t ckpt,
                 const float** out, std::string& err);

    /// the prompt rows' width and longest chunk without a device side (init_device sets both; the tests use this)
    void set_rows_geometry(int64_t d, int64_t max_chunk);

    bool reset(std::string& err);                        ///< every worker's session to position 0
    bool ckpt_save(uint64_t id, std::string& err);       ///< every worker keeps its running state under id
    bool ckpt_restore(uint64_t id, std::string& err);    ///< ... and puts it back
    bool ckpt_retain(const std::vector<uint64_t>& ids, std::string& err);   ///< drop every other id
    bool end_request(std::string& err);                  ///< a request ended (workers refill, log)
    uint64_t new_ckpt_id() { return ++ckpt_seq_; }
    void bye();                                          ///< tell the workers we are leaving (they stay loaded)

    /// one line for the log / INFO: per worker its range, cache and traffic
    std::string summary() const;
    int64_t worker_slots() const;
    /// After end_request: one log line per request that splits a window's time into this PC, each worker's layers
    /// and the network (from the workers' own timings), then resets the counters.  `decode_ms`: the request's decode.
    void report_request(double decode_ms);
    double ms_net = 0;          ///< wall time spent waiting on workers (verify windows)
    double ms_prefill_net = 0;  ///< ... (prompt chunks)
    int64_t windows = 0;

private:
    bool connect_one(const std::string& addr, Channel& out, KV& hello, std::string& err);
    KV config_for(size_t i) const;
    bool forward(Msg type, const std::vector<uint8_t>& prefix_tmpl, int64_t T, int64_t floats_per_row,
                 const uint8_t* first, Wire last_wire, std::vector<uint8_t>& last, std::string& err);
    std::vector<WorkerInfo> workers_;
    std::vector<Channel> ch_;
    std::mutex mu_;
    Wire wire_ = Wire::F32, draft_wire_ = Wire::F16;
    double timeout_s_ = 300.0, wait_s_ = 900.0;
    std::string secret_;
    ModelFingerprint fp_;
    KV common_;
    uint64_t ckpt_seq_ = 0;
    uint64_t req_bytes0_ = 0;   ///< bytes on all channels when the request's counters were last reset

    // the window and the head (device)
    const core::WeightTable* wt_ = nullptr;
    const core::ModelGeometry* g_ = nullptr;
    core::BlockBuffers block_{};
    const core::NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int64_t n_vocab_ = 0, hb_ = 0, d_ = 0, max_chunk_ = 0;
    const float* hand_host_ = nullptr;   ///< the coordinator verifier's hand-off (mapped host)
    float* R_dev_ = nullptr;             ///< (max_t, hc*n_embd): the final residual of the last window
    float* mixed_dev_ = nullptr;
    uint8_t* xq_dev_ = nullptr;
    float* logits_dev_ = nullptr;
    int32_t* out_host_ = nullptr;        ///< mapped
    int32_t* out_dev_ = nullptr;
    float* rows_host_ = nullptr;         ///< pinned: decoded window rows (max_t x hb)
    float* pf_host_ = nullptr;           ///< pinned: decoded prompt rows (max_chunk x d)
    void* stream_ = nullptr;
    std::vector<uint8_t> enc_, buf_a_, buf_b_, fin_;
    std::vector<float> pf_vec_;
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();
    const int32_t* hist_d_ = nullptr;
    int hist_len_ = 0;
    bool head_sampling_ = true;
};

}  // namespace strata::pool
