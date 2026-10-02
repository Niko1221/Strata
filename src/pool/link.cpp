// src/pool/link.cpp - the pool's coordinator: the workers' connections, the window's head, the prompt rows.
#include "strata/pool/link.hpp"

#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>

namespace strata::pool {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

// ------------------------------------------------------------------------------------------------ fingerprint
KV ModelFingerprint::to_kv() const {
    KV kv;
    kv.set("n_layers", n_layers);
    kv.set("n_expert", n_expert);
    kv.set("k", k);
    kv.set("n_embd", n_embd);
    kv.set("hc", hc);
    kv.set("experts_bytes", (int64_t) experts_bytes);
    kv.set("dense_bytes", (int64_t) dense_bytes);
    kv.set("native", (int64_t) native);
    kv.set("pack", pack);
    kv.set("engine", engine);
    return kv;
}

std::string ModelFingerprint::mismatch(const KV& w) const {
    std::string d;
    auto cmp = [&](const char* key, int64_t mine) {
        const int64_t theirs = w.i64(key, -1);
        if (theirs != mine)
            d += std::string(d.empty() ? "" : ", ") + key + " " + std::to_string(theirs) + " (here " +
                 std::to_string(mine) + ")";
    };
    cmp("n_layers", n_layers);
    cmp("n_expert", n_expert);
    cmp("k", k);
    cmp("n_embd", n_embd);
    cmp("hc", hc);
    cmp("experts_bytes", (int64_t) experts_bytes);
    cmp("native", native);
    if (!d.empty())
        d = "it runs a different model (" + d + "; its pack " + w.str("pack", "?") + ", here " + pack +
            ") - install the same model and size on both PCs";
    return d;
}

PoolLink::~PoolLink() {
    bye();   // the workers keep their layers loaded for the next coordinator
    if (stream_) cudaStreamDestroy((cudaStream_t) stream_);
    for (void* p : {(void*) R_dev_, (void*) mixed_dev_, (void*) xq_dev_, (void*) logits_dev_})
        if (p) cudaFree(p);
    for (void* p : {(void*) out_host_, (void*) rows_host_, (void*) pf_host_})
        if (p) cudaFreeHost(p);
}

// ------------------------------------------------------------------------------------------------ connect
bool PoolLink::connect_one(const std::string& addr, Channel& out, KV& hello, std::string& err) {
    std::string host;
    int port = kDefaultPort;
    if (!split_host_port(addr, kDefaultPort, host, port)) { err = "bad pool peer address '" + addr + "'"; return false; }
    const auto t0 = Clock::now();
    auto last_note = t0;
    for (;;) {
        std::string ce;
        Socket s = connect_to(host, port, 5.0, ce);
        if (s.valid()) {
            s.set_timeout(60.0);
            Channel ch(std::move(s));
            Msg t;
            std::vector<uint8_t> p;
            if (!ch.recv(t, p, err, 1 << 20)) { err = "pool worker " + addr + ": " + err; return false; }
            if (t == Msg::Err) {
                const KV e = KV::decode(p);
                if (e.i64("busy", 0) != 0 && ms_since(t0) < wait_s_ * 1000.0) {   // serving another coordinator
                    if (ms_since(last_note) > 10000.0) {
                        last_note = Clock::now();
                        std::fprintf(stderr, "strata pool: worker %s is busy (%s); waiting ...\n", addr.c_str(),
                                     e.str("message").c_str());
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    continue;
                }
                err = "pool worker " + addr + ": " + e.str("message", "refused");
                return false;
            }
            if (t != Msg::Hello) { err = "pool worker " + addr + ": " + unexpected(t, Msg::Hello, p); return false; }
            hello = KV::decode(p);
            if (hello.i64("proto", 0) != kProtocol) {
                err = "pool worker " + addr + " speaks protocol " + hello.str("proto", "?") + ", this engine " +
                      std::to_string(kProtocol) + ": run the same Strata version on every PC";
                return false;
            }
            if (const std::string mm = fp_.mismatch(hello); !mm.empty()) { err = "pool worker " + addr + ": " + mm; return false; }
            // the secret, both ways: the worker proves it knows it too, so a stranger cannot pose as one
            const std::string wn = hello.str("nonce");
            const std::string cn = random_nonce();
            const auto mac = hmac_sha256(secret_, wn.data(), wn.size());
            KV auth;
            auth.set("mac", to_hex(mac.data(), mac.size()));
            auth.set("nonce", cn);
            auth.set("engine", fp_.engine);
            if (!ch.send(Msg::Auth, auth, err) || !ch.recv(t, p, err, 1 << 20)) {
                err = "pool worker " + addr + ": " + err;
                return false;
            }
            if (t != Msg::AuthOk) { err = "pool worker " + addr + ": " + unexpected(t, Msg::AuthOk, p); return false; }
            const auto want = hmac_sha256(secret_, cn.data(), cn.size());
            if (!equal_ct(KV::decode(p).str("mac"), to_hex(want.data(), want.size()))) {
                err = "pool worker " + addr + " does not know this pool's secret (set the same pool secret on every PC)";
                return false;
            }
            ch.socket().set_timeout(timeout_s_);
            out = std::move(ch);
            return true;
        }
        if (ms_since(t0) > wait_s_ * 1000.0) {
            err = "pool worker " + addr + " is not reachable: " + ce +
                  " (is Strata running there as a pool worker, and does its firewall allow the port?)";
            return false;
        }
        if (ms_since(last_note) > 10000.0) {
            last_note = Clock::now();
            std::fprintf(stderr, "strata pool: waiting for worker %s (%s) ...\n", addr.c_str(), ce.c_str());
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

bool PoolLink::connect(const std::vector<std::string>& peers, const std::string& secret, const ModelFingerprint& fp,
                       double wait_s, double timeout_s, std::string& err) {
    timeout_s_ = timeout_s;
    wait_s_ = wait_s;
    secret_ = secret;
    fp_ = fp;
    workers_.clear();
    ch_.clear();
    for (const std::string& addr : peers) {
        WorkerInfo w;
        w.addr = addr;
        Channel ch;
        if (!connect_one(addr, ch, w.hello, err)) return false;
        std::fprintf(stderr, "strata pool: worker %s (%s, %s, %.1f GiB VRAM free, %.1f GiB RAM)%s connected\n",
                     addr.c_str(), w.hello.str("name", "?").c_str(), w.hello.str("gpu", "?").c_str(),
                     (double) w.hello.i64("vram_free", 0) / 1073741824.0,
                     (double) w.hello.i64("ram_total", 0) / 1073741824.0,
                     w.hello.i64("loaded", 0) ? (" (layers " + w.hello.str("loaded_lb") + "-" +
                                                 std::to_string(w.hello.i64("loaded_le") - 1) + " loaded)").c_str() : "");
        workers_.push_back(std::move(w));
        ch_.push_back(std::move(ch));
    }
    return true;
}

KV PoolLink::config_for(size_t i) const {
    const WorkerInfo& w = workers_[i];
    KV c = common_;
    c.set("lb", w.lb);
    c.set("le", w.le);
    c.set("index", (int64_t) i);
    c.set("last", (int64_t) (i + 1 == workers_.size()));
    c.set("wire", std::string(wire_name(wire_)));
    c.set("draft_wire", std::string(wire_name(draft_wire_)));
    return c;
}

bool PoolLink::configure(const std::vector<int64_t>& at, int64_t n_layers, const KV& common, Wire wire,
                         Wire draft_wire, std::string& err) {
    if (at.size() != workers_.size()) { err = "pool: one split point per worker is needed"; return false; }
    common_ = common;
    wire_ = wire;
    draft_wire_ = draft_wire;
    for (size_t i = 0; i < workers_.size(); ++i) {
        WorkerInfo& w = workers_[i];
        w.lb = at[i];
        w.le = i + 1 < at.size() ? at[i + 1] : n_layers;
        if (!ch_[i].send(Msg::Config, config_for(i), err)) { err = "pool worker " + w.addr + ": " + err; return false; }
    }
    return true;
}

bool PoolLink::wait_ready(double timeout_s, std::string& err) {
    for (size_t i = 0; i < workers_.size(); ++i) {
        WorkerInfo& w = workers_[i];
        for (int attempt = 0;; ++attempt) {
            ch_[i].socket().set_timeout(timeout_s);
            Msg t;
            std::vector<uint8_t> p;
            if (!ch_[i].recv(t, p, err, 1 << 20)) {
                err = "pool worker " + w.addr + " did not get ready: " + err;
                return false;
            }
            if (t == Msg::Err && KV::decode(p).i64("reload", 0) != 0 && attempt < 2) {
                // it had another range loaded: it restarts to load ours, then takes the connection again
                std::fprintf(stderr, "strata pool: worker %s reloads for layers %lld-%lld ...\n", w.addr.c_str(),
                             (long long) w.lb, (long long) (w.le - 1));
                ch_[i].close();
                Channel ch;
                KV hello;
                if (!connect_one(w.addr, ch, hello, err)) return false;
                ch_[i] = std::move(ch);
                w.hello = hello;
                if (!ch_[i].send(Msg::Config, config_for(i), err)) { err = "pool worker " + w.addr + ": " + err; return false; }
                continue;
            }
            if (t != Msg::Ready) { err = "pool worker " + w.addr + ": " + unexpected(t, Msg::Ready, p); return false; }
            w.ready = KV::decode(p);
            break;
        }
        ch_[i].socket().set_timeout(timeout_s_);
        std::fprintf(stderr, "strata pool: worker %s ready: layers %lld-%lld, %lld experts in its VRAM (%.2f GiB), "
                             "%.1f GiB of experts in its RAM, %lld MiB VRAM free, prompt chunk %lld\n",
                     w.addr.c_str(), (long long) w.lb, (long long) (w.le - 1), (long long) w.ready.i64("slots"),
                     (double) w.ready.i64("cache_mib") / 1024.0, (double) w.ready.i64("arena_mib") / 1024.0,
                     (long long) w.ready.i64("vram_free_mib"), (long long) w.ready.i64("chunk"));
    }
    return true;
}

int64_t PoolLink::max_chunk() const {
    int64_t c = 0;
    for (const WorkerInfo& w : workers_) {
        const int64_t k = w.ready.i64("chunk", 0);
        if (k > 0) c = c == 0 ? k : std::min(c, k);
    }
    return c;
}

int64_t PoolLink::worker_slots() const {
    int64_t n = 0;
    for (const WorkerInfo& w : workers_) n += w.ready.i64("slots", 0);
    return n;
}

std::string PoolLink::summary() const {
    std::string s;
    for (size_t i = 0; i < workers_.size(); ++i) {
        const WorkerInfo& w = workers_[i];
        char b[256];
        std::snprintf(b, sizeof b, "%s%s=%lld-%lld/%lld", s.empty() ? "" : ",", w.addr.c_str(), (long long) w.lb,
                      (long long) (w.le - 1), (long long) w.ready.i64("slots"));
        s += b;
    }
    return s;
}

// ------------------------------------------------------------------------------------------------ device side
bool PoolLink::init_device(const core::WeightTable& wt, const core::ModelGeometry& g, const core::BlockBuffers& block,
                           const core::NativeHead* head, int max_t, int64_t n_vocab, int64_t max_chunk,
                           const float* handoff_host, std::string& err) {
    wt_ = &wt;
    g_ = &g;
    block_ = block;
    head_ = head;
    max_t_ = max_t;
    n_vocab_ = n_vocab;
    hb_ = core::Verifier::handoff_floats(g);
    d_ = g.hc * g.n_embd;
    max_chunk_ = max_chunk;
    hand_host_ = handoff_host;
    const size_t T = (size_t) max_t, N = (size_t) g.n_embd;
    cudaStream_t s = nullptr;
    bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
    stream_ = s;
    ok = ok && cudaMalloc((void**) &R_dev_, T * (size_t) d_ * 4) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &mixed_dev_, T * N * 4) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &xq_dev_, strata::kernels::native_q8_1_bytes((int) N, (int) T)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &logits_dev_, T * (size_t) n_vocab * 4) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &out_host_, (T + 4) * 4, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &out_dev_, out_host_, 0) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &rows_host_, T * (size_t) hb_ * 4, cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &pf_host_, (size_t) std::max<int64_t>(max_chunk, 1) * (size_t) d_ * 4,
                             cudaHostAllocDefault) == cudaSuccess;
    if (!ok) {
        err = std::string("pool: the head's buffers on this GPU: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    cudaMemset(R_dev_, 0, T * (size_t) d_ * 4);
    return true;
}

// ------------------------------------------------------------------------------------------------ the hops
// `prefix_tmpl` is the message's binary prefix (the same for every worker); `first` are the rows in the wire format
// (T x floats_per_row).  Each worker's reply rows go to the next worker; the last reply's bytes end in `last` (in
// `last_wire`, which the last worker was told to use for this kind of message).
bool PoolLink::forward(Msg type, const std::vector<uint8_t>& prefix_tmpl, int64_t T, int64_t floats_per_row,
                       const uint8_t* first, Wire last_wire, std::vector<uint8_t>& last, std::string& err) {
    const Msg want = type == Msg::Verify ? Msg::VerifyRows : Msg::PrefillRows;
    const uint8_t* cur = first;
    size_t cur_bytes = (size_t) T * (size_t) floats_per_row * wire_bytes(wire_);
    std::vector<uint8_t>* bufs[2] = {&buf_a_, &buf_b_};
    for (size_t i = 0; i < ch_.size(); ++i) {
        Channel& c = ch_[i];
        if (!c.drain(err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        if (!c.send(type, prefix_tmpl.data(), prefix_tmpl.size(), cur, cur_bytes, err)) {
            err = "pool worker " + workers_[i].addr + ": " + err;
            return false;
        }
        std::vector<uint8_t>& in = i + 1 == ch_.size() ? last : *bufs[i & 1];
        Msg t;
        if (!c.recv(t, in, err, (uint64_t) 8 << 30)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        if (t != want) { err = "pool worker " + workers_[i].addr + ": " + unexpected(t, want, in); return false; }
        const Wire w = i + 1 == ch_.size() ? last_wire : wire_;
        const size_t expect = (size_t) T * (size_t) floats_per_row * wire_bytes(w);
        if (in.size() != expect) {
            err = "pool worker " + workers_[i].addr + " returned " + std::to_string(in.size()) + " bytes of rows, not " +
                  std::to_string(expect);
            return false;
        }
        cur = in.data();
        cur_bytes = in.size();
    }
    return true;
}

bool PoolLink::run(int T, const int32_t* tokens, int64_t pos0, int32_t* out, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (T < 1 || T > max_t_) { err = "pool: window size out of range"; return false; }
    const auto t0 = Clock::now();
    Packer pk;
    pk.put<int32_t>(T);
    pk.put<int64_t>(pos0);
    pk.put_bytes(tokens, (size_t) T * 4);
    const size_t n = (size_t) T * (size_t) hb_;
    const uint8_t* first = (const uint8_t*) hand_host_;
    if (wire_ != Wire::F32) {
        enc_.resize(n * wire_bytes(wire_));
        encode_rows(wire_, hand_host_, n, enc_.data());
        first = enc_.data();
    }
    if (!forward(Msg::Verify, pk.b, T, hb_, first, wire_, fin_, err)) return false;
    decode_rows(wire_, fin_.data(), n, rows_host_);
    ms_net += ms_since(t0);
    ++windows;
    const cudaStream_t cs = (cudaStream_t) stream_;
    // the residual part of every row (the pending write and inject are a later stage's business; the last layer
    // has applied its write already)
    if (cudaMemcpy2DAsync(R_dev_, (size_t) d_ * 4, rows_host_, (size_t) hb_ * 4, (size_t) d_ * 4, (size_t) T,
                          cudaMemcpyHostToDevice, cs) != cudaSuccess) {
        err = std::string("pool: the final rows' upload: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (!head_sampling_) {   // a prompt read through windows: the picks are discarded, only the rows matter
        if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "pool: the final rows' upload failed"; return false; }
        for (int t = 0; t < T; ++t) out[t] = 0;
        return true;
    }
    const int64_t N = g_->n_embd;
    for (int t = 0; t < T; ++t) {
        core::BlockBuffers bb = block_;
        bb.R = R_dev_ + (size_t) t * (size_t) d_;
        bb.mixed = mixed_dev_ + (size_t) t * (size_t) N;
        if (head_ != nullptr && head_->loaded()) {
            if (!core::lm_head_mix(*wt_, *g_, bb, cs, err)) return false;
        } else if (!core::lm_head(*wt_, *g_, bb, logits_dev_ + (size_t) t * (size_t) n_vocab_, cs, err)) {
            return false;
        }
    }
    if (head_ != nullptr && head_->loaded()) {
        try {
            strata::kernels::native_quantize_q8_1(mixed_dev_, xq_dev_, (int) N, T, cs);
            strata::kernels::native_mmvq(head_->type(), head_->weights(), xq_dev_, logits_dev_, (int) N, (int) n_vocab_,
                                         T, cs);
        } catch (const std::exception& e) {
            err = std::string("pool head: ") + e.what();
            return false;
        }
    }
    // the verify window's own rule: greedy, or this request's sampling with Philox(seed, pos0 + t) and penalties
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    strata::kernels::SamplerParams sp;
    if (sampled || hist_d_ != nullptr) {
        sp = sampling_;
        sp.counter = (uint64_t) pos0;
        strata::kernels::sample_tokens(logits_dev_, T, (int) n_vocab_, hist_d_, hist_len_, sp, out_dev_, cs);
    } else {
        sp.greedy = true;
        sp.temperature = 0.0f;
        strata::kernels::sample_tokens(logits_dev_, T, (int) n_vocab_, nullptr, 0, sp, out_dev_, cs);
    }
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        err = std::string("pool: the head: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) out_host_)[t];
    return true;
}

bool PoolLink::commit(int n_keep, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    Packer pk;
    pk.put<int32_t>(n_keep);
    for (size_t i = 0; i < ch_.size(); ++i) {
        if (!ch_[i].send(Msg::Commit, pk.b, err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        ++ch_[i].pending;
    }
    return true;
}

void PoolLink::set_rows_geometry(int64_t d, int64_t max_chunk) {
    d_ = d;
    max_chunk_ = max_chunk;
}

bool PoolLink::prefill(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, int64_t segment, uint64_t ckpt,
                       const float** out, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (T < 1 || T > max_chunk_) { err = "pool: a prompt chunk of " + std::to_string(T) + " tokens is too long"; return false; }
    const auto t0 = Clock::now();
    Packer pk;
    pk.put<int64_t>(p0);
    pk.put<int64_t>(T);
    pk.put<int64_t>(segment);
    pk.put<uint64_t>(ckpt);
    pk.put_bytes(tokens, (size_t) T * 8);
    const size_t n = (size_t) T * (size_t) d_;
    const uint8_t* first = (const uint8_t*) rows;
    if (wire_ != Wire::F32) {
        enc_.resize(n * wire_bytes(wire_));
        encode_rows(wire_, rows, n, enc_.data());
        first = enc_.data();
    }
    if (!forward(Msg::Prefill, pk.b, T, d_, first, draft_wire_, fin_, err)) return false;
    float* dst = pf_host_;
    if (dst == nullptr) {   // no device side (the tests): plain memory
        pf_vec_.resize(n);
        dst = pf_vec_.data();
    }
    decode_rows(draft_wire_, fin_.data(), n, dst);
    *out = dst;
    ms_prefill_net += ms_since(t0);
    return true;
}

// ------------------------------------------------------------------------------------------------ session ops
bool PoolLink::reset(std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < ch_.size(); ++i) {
        if (!ch_[i].send(Msg::Reset, err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        ++ch_[i].pending;
    }
    return true;
}

bool PoolLink::ckpt_save(uint64_t id, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    Packer pk;
    pk.put<uint64_t>(id);
    for (size_t i = 0; i < ch_.size(); ++i) {
        if (!ch_[i].send(Msg::CkptSave, pk.b, err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        ++ch_[i].pending;
    }
    return true;
}

bool PoolLink::ckpt_restore(uint64_t id, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    Packer pk;
    pk.put<uint64_t>(id);
    for (size_t i = 0; i < ch_.size(); ++i) {
        Channel& c = ch_[i];
        Msg t;
        std::vector<uint8_t> p;
        if (!c.drain(err) || !c.send(Msg::CkptRestore, pk.b, err) || !c.recv(t, p, err, 1 << 20)) {
            err = "pool worker " + workers_[i].addr + ": " + err;
            return false;
        }
        if (t != Msg::Ack) { err = "pool worker " + workers_[i].addr + ": " + unexpected(t, Msg::Ack, p); return false; }
    }
    return true;
}

bool PoolLink::ckpt_retain(const std::vector<uint64_t>& ids, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    Packer pk;
    pk.put<uint32_t>((uint32_t) ids.size());
    for (const uint64_t id : ids) pk.put<uint64_t>(id);
    for (size_t i = 0; i < ch_.size(); ++i) {
        if (!ch_[i].send(Msg::CkptRetain, pk.b, err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        ++ch_[i].pending;
    }
    return true;
}

bool PoolLink::end_request(std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < ch_.size(); ++i) {
        if (!ch_[i].drain(err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        const auto t0 = Clock::now();
        ch_[i].last_ack.clear();
        if (!ch_[i].send(Msg::EndRequest, err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        ++ch_[i].pending;
        if (!ch_[i].drain(err)) { err = "pool worker " + workers_[i].addr + ": " + err; return false; }
        workers_[i].ms_rtt = ms_since(t0);
        workers_[i].req = ch_[i].last_ack.empty() ? KV() : KV::decode(ch_[i].last_ack);
    }
    return true;
}

void PoolLink::report_request(double decode_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t bytes = 0;
    for (const Channel& c : ch_) bytes += c.bytes_out + c.bytes_in;
    if (windows > 0) {
        const double w = (double) windows;
        double layers_ms = 0;   // the workers' own time per window, summed (they run one after the other)
        bool all = true;
        std::string per;
        for (const WorkerInfo& wi : workers_) {
            const int64_t ww = wi.req.i64("windows", 0);
            if (ww <= 0) { all = false; continue; }
            const double ms = wi.req.f64("ms_verify", 0) / (double) ww;
            const int64_t look = wi.req.i64("lookups", 0);
            layers_ms += ms;
            char b[256];
            std::snprintf(b, sizeof b, "; %s: layers %lld-%lld %.1f ms, its expert cache hit rate %.1f%%, %lld experts in "
                          "its VRAM", wi.addr.c_str(), (long long) wi.lb, (long long) wi.le - 1, ms,
                          look > 0 ? 100.0 * (double) wi.req.i64("hits", 0) / (double) look : 0.0,
                          (long long) wi.ready.i64("slots", 0));
            per += b;
        }
        const double net = ms_net / w;
        const double here = decode_ms / w - net;
        if (all) {
            std::fprintf(stderr, "strata pool: %lld windows, %.1f ms each = this PC %.1f ms + the workers' layers %.1f ms + "
                                 "the network %.1f ms (%.0f KiB a window, %.1f ms round trip)%s\n",
                         (long long) windows, decode_ms / w, here, layers_ms, net - layers_ms,
                         (double) (bytes - req_bytes0_) / w / 1024.0, workers_.empty() ? 0.0 : workers_[0].ms_rtt,
                         per.c_str());
        } else {
            std::fprintf(stderr, "strata pool: %lld windows, %.1f ms each = this PC %.1f ms + the workers and the network "
                                 "%.1f ms%s\n", (long long) windows, decode_ms / w, here, net, per.c_str());
        }
    }
    if (ms_prefill_net > 0)
        std::fprintf(stderr, "strata pool: prompt chunks waited %.0f ms on the workers\n", ms_prefill_net);
    windows = 0;
    ms_net = ms_prefill_net = 0;
    req_bytes0_ = bytes;
}

void PoolLink::bye() {
    std::lock_guard<std::mutex> lk(mu_);
    for (Channel& c : ch_) {
        if (!c.valid()) continue;
        std::string e;
        c.drain(e);
        c.send(Msg::Bye, e);
        c.close();
    }
}

}  // namespace strata::pool
