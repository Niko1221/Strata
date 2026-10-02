// src/pool/pool_link_test.cpp - the coordinator's PoolLink against scripted workers, without a GPU: the handshake
// (secret, model check, busy, reload), CONFIG / READY, a prompt chunk through two workers in both wire formats, the
// lazily acknowledged commits and resets, and checkpoints.  (A verify window needs the GPU head: not here.)
#include "strata/pool/link.hpp"
#include "strata/pool/worker.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

using namespace strata::pool;

static int failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);        \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static ModelFingerprint fp() {
    ModelFingerprint f;
    f.n_layers = 48; f.n_expert = 256; f.k = 10; f.n_embd = 2560; f.hc = 4;
    f.experts_bytes = 123456789; f.dense_bytes = 42; f.native = 1; f.pack = "coder-iq1_m"; f.engine = "0.1.30";
    return f;
}

// A worker: adds `add` to every row value it gets (so the coordinator can tell who ran), keeps checkpoints by id.
struct FakeWorker {
    Listener l;
    std::thread t;
    std::string secret;
    float add;
    std::atomic<int> commits{0}, resets{0}, prefills{0}, sessions{0};
    std::set<uint64_t> ckpts;
    std::mutex mu;
    bool reload_first = false;     // answer the first CONFIG with "reload" (it had another range)
    bool wrong_model = false;
    int port = 0;
    std::atomic<bool> stop{false};

    FakeWorker(std::string s, float a) : secret(std::move(s)), add(a) {
        std::string e;
        l.open("127.0.0.1", 0, e);
        port = l.port();
    }
    void run(int sessions_to_serve) {
        t = std::thread([this, sessions_to_serve] {
            for (int sess = 0; sess < sessions_to_serve; ++sess) {
                Channel ch;
                KV cfg;
                auto hello = [&] {
                    KV h = fp().to_kv();
                    if (wrong_model) h.set("n_expert", (int64_t) 512);
                    h.set("proto", (int64_t) kProtocol);
                    h.set("name", std::string("fake"));
                    h.set("vram_free", (int64_t) 8 << 30);
                    return h;
                };
                if (!accept_coordinator(l, secret, hello, ch, cfg, [](const std::string&) {}, &stop)) return;
                ++sessions;
                if (reload_first) {
                    reload_first = false;
                    KV no;
                    no.set("reload", (int64_t) 1);
                    no.set("message", std::string("reloading"));
                    std::string e;
                    ch.send(Msg::Err, no, e);
                    ch.close();
                    continue;   // "restarts" and takes the connection again
                }
                const Wire wire = cfg.str("wire") == "f16" ? Wire::F16 : Wire::F32;
                const Wire dwire = cfg.i64("last") ? (cfg.str("draft_wire") == "bf16" ? Wire::BF16 : cfg.str("draft_wire") == "f16" ? Wire::F16 : Wire::F32) : wire;
                KV ready;
                ready.set("slots", (int64_t) 1000);
                ready.set("chunk", (int64_t) 64);
                ready.set("cache_mib", (int64_t) 2048);
                std::string e;
                ch.send(Msg::Ready, ready, e);
                std::vector<uint8_t> p;
                for (;;) {
                    Msg t;
                    if (!ch.recv(t, p, e)) break;
                    Unpacker u(p.data(), p.size());
                    if (t == Msg::Prefill) {
                        int64_t p0, T, seg;
                        uint64_t ck;
                        u.get(p0); u.get(T); u.get(seg); u.get(ck);
                        u.skip((size_t) T * 8);
                        const size_t n = u.left() / wire_bytes(wire);
                        std::vector<float> rows(n);
                        decode_rows(wire, u.here(), n, rows.data());
                        for (float& x : rows) x += add;
                        std::vector<uint8_t> out(n * wire_bytes(dwire));
                        encode_rows(dwire, rows.data(), n, out.data());
                        if (ck) { std::lock_guard<std::mutex> g(mu); ckpts.insert(ck); }
                        ++prefills;
                        ch.send(Msg::PrefillRows, out, e);
                    } else if (t == Msg::Commit) {
                        ++commits;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::Reset) {
                        ++resets;
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::CkptSave) {
                        uint64_t id; u.get(id);
                        { std::lock_guard<std::mutex> g(mu); ckpts.insert(id); }
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::CkptRestore) {
                        uint64_t id; u.get(id);
                        bool have;
                        { std::lock_guard<std::mutex> g(mu); have = ckpts.count(id) != 0; }
                        if (have) ch.send(Msg::Ack, e);
                        else { KV no; no.set("message", "no checkpoint " + std::to_string(id)); ch.send(Msg::Err, no, e); }
                    } else if (t == Msg::CkptRetain) {
                        uint32_t n; u.get(n);
                        std::set<uint64_t> keep;
                        for (uint32_t i = 0; i < n; ++i) { uint64_t id; u.get(id); keep.insert(id); }
                        { std::lock_guard<std::mutex> g(mu); std::set<uint64_t> k2; for (auto id : ckpts) if (keep.count(id)) k2.insert(id); ckpts = k2; }
                        ch.send(Msg::Ack, e);
                    } else if (t == Msg::EndRequest) {
                        KV st;
                        st.set("windows", (int64_t) 4);
                        st.set("ms_verify", 10.0);
                        st.set("hits", (int64_t) 3);
                        st.set("lookups", (int64_t) 4);
                        ch.send(Msg::Ack, st, e);
                    } else if (t == Msg::Bye) {
                        break;
                    }
                }
            }
        });
    }
    ~FakeWorker() {
        stop = true;
        if (t.joinable()) t.join();
    }
};

static void test_two_workers(Wire wire) {
    FakeWorker a("pw", 1.0f), b("pw", 10.0f);
    a.run(1);
    b.run(1);
    PoolLink link;
    std::string err;
    const std::vector<std::string> peers = {"127.0.0.1:" + std::to_string(a.port), "127.0.0.1:" + std::to_string(b.port)};
    CHECK(link.connect(peers, "pw", fp(), 10.0, 10.0, err));
    CHECK(link.size() == 2);
    KV common;
    common.set("max_context", (int64_t) 4096);
    CHECK(link.configure({20, 34}, 48, common, wire, Wire::F32, err));
    CHECK(link.workers()[0].lb == 20 && link.workers()[0].le == 34 && link.workers()[1].le == 48);
    CHECK(link.wait_ready(10.0, err));
    CHECK(link.max_chunk() == 64 && link.worker_slots() == 2000);
    // a prompt chunk: 3 rows of 8 floats through both workers (+1 then +10)
    link.set_rows_geometry(8, 64);
    std::vector<float> rows(24);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) i * 0.25f;
    const int64_t toks[3] = {1, 2, 3};
    const float* out = nullptr;
    CHECK(link.prefill(rows.data(), toks, 3, 0, 3, 7, &out, err));
    for (size_t i = 0; out && i < rows.size(); ++i) CHECK(std::fabs(out[i] - (rows[i] + 11.0f)) < 1e-2f);
    // commits are acknowledged lazily: many in a row, then a synchronous restore drains them all
    for (int i = 0; i < 25; ++i) CHECK(link.commit(1, err));
    CHECK(link.reset(err));
    CHECK(link.ckpt_save(99, err));
    CHECK(link.ckpt_restore(99, err));
    CHECK(link.ckpt_restore(7, err));                       // saved with the prompt chunk
    CHECK(link.ckpt_retain({99}, err));
    CHECK(!link.ckpt_restore(7, err) && err.find("no checkpoint 7") != std::string::npos);
    err.clear();
    CHECK(link.end_request(err));
    CHECK(link.workers()[0].req.i64("windows") == 4 && link.workers()[1].req.f64("ms_verify") == 10.0);
    link.windows = 4;
    link.ms_net = 60.0;
    link.report_request(400.0);
    link.bye();
    CHECK(a.commits == 25 && b.commits == 25 && a.resets == 1 && a.prefills == 1 && b.prefills == 1);
}

static void test_refusals() {
    std::string err;
    {   // the wrong secret
        FakeWorker w("right", 0);
        w.run(1);
        PoolLink link;
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(w.port)}, "wrong", fp(), 5.0, 5.0, err));
        CHECK(err.find("secret") != std::string::npos);
        // let the fake worker's accept end: a coordinator with the right secret that leaves at once
        PoolLink ok;
        std::string e2;
        CHECK(ok.connect({"127.0.0.1:" + std::to_string(w.port)}, "right", fp(), 5.0, 5.0, e2));
        KV c;
        CHECK(ok.configure({24}, 48, c, Wire::F32, Wire::F16, e2) && ok.wait_ready(5.0, e2));
    }
    {   // another model on the worker
        FakeWorker w("s", 0);
        w.wrong_model = true;
        w.run(1);
        PoolLink link;
        err.clear();
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 5.0, 5.0, err));
        CHECK(err.find("different model") != std::string::npos);
        // unblock the fake worker's accept (it is waiting for a CONFIG on that connection: closing ends it)
    }
    {   // nobody there: a clear error after the wait
        Listener tmp;
        tmp.open("127.0.0.1", 0, err);
        const int dead = tmp.port();
        tmp.close();
        PoolLink link;
        err.clear();
        CHECK(!link.connect({"127.0.0.1:" + std::to_string(dead)}, "s", fp(), 1.0, 1.0, err));
        CHECK(err.find("not reachable") != std::string::npos);
    }
}

static void test_reload_and_busy() {
    std::string err;
    FakeWorker w("s", 0);
    w.reload_first = true;
    w.run(2);   // the first session answers "reload"; the second serves
    PoolLink link;
    CHECK(link.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 10.0, 10.0, err));
    KV c;
    CHECK(link.configure({30}, 48, c, Wire::F32, Wire::F16, err));
    CHECK(link.wait_ready(10.0, err));
    CHECK(w.sessions == 2);
    // while it serves, a second coordinator hears "busy" and gives up after its wait
    {
        BusyResponder busy(w.l, "test");
        PoolLink other;
        std::string e2;
        CHECK(!other.connect({"127.0.0.1:" + std::to_string(w.port)}, "s", fp(), 2.0, 2.0, e2));
        CHECK(e2.find("serving") != std::string::npos);
    }
    link.bye();
}

int main() {
    net_init();
    test_two_workers(Wire::F32);
    test_two_workers(Wire::F16);
    test_refusals();
    test_reload_and_busy();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("strata-pool-link-test: all checks passed\n");
    return 0;
}
