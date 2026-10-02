// src/pool/pool_test.cpp - the pool's network pieces without a GPU: hashing, the wire formats, frames over a
// loopback socket, the worker handshake (both secrets), the busy answer, and the split search.
#include "strata/pool/net.hpp"
#include "strata/pool/protocol.hpp"
#include "strata/pool/split.hpp"
#include "strata/pool/worker.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
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

static void test_hashes() {
    const auto d = sha256("abc", 3);
    CHECK(to_hex(d.data(), d.size()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const auto e = sha256("", 0);
    CHECK(to_hex(e.data(), e.size()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    std::string m(1000, 'a');
    const auto big = sha256(m.data(), m.size());   // crosses many blocks
    CHECK(to_hex(big.data(), big.size()) == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    // RFC 4231, test case 2
    const std::string data = "what do ya want for nothing?";
    const auto h = hmac_sha256("Jefe", data.data(), data.size());
    CHECK(to_hex(h.data(), h.size()) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    // RFC 4231, test case 6 (a key longer than the block)
    const std::string k6(131, '\xaa');
    const std::string d6 = "Test Using Larger Than Block-Size Key - Hash Key First";
    const auto h6 = hmac_sha256(k6, d6.data(), d6.size());
    CHECK(to_hex(h6.data(), h6.size()) == "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    CHECK(random_nonce().size() == 32 && random_nonce() != random_nonce());
    CHECK(equal_ct("abc", "abc") && !equal_ct("abc", "abd") && !equal_ct("abc", "ab"));
}

static void test_wire() {
    const float special[] = {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 1e-5f, 6.0e-8f, 3.14159265f, 1e30f, -1e-30f,
                             INFINITY, -INFINITY};
    const size_t n = sizeof special / sizeof special[0];
    for (Wire w : {Wire::F32, Wire::F16, Wire::BF16}) {
        std::vector<uint8_t> b(n * wire_bytes(w));
        std::vector<float> back(n);
        encode_rows(w, special, n, b.data());
        decode_rows(w, b.data(), n, back.data());
        for (size_t i = 0; i < n; ++i) {
            const float x = special[i], y = back[i];
            if (w == Wire::F32) { CHECK(std::memcmp(&x, &y, 4) == 0); continue; }
            if (std::isinf(x)) { CHECK(std::isinf(y) && (x > 0) == (y > 0)); continue; }
            if (w == Wire::F16 && std::fabs(x) > 65504.0f) { CHECK(std::isinf(y)); continue; }
            if (w == Wire::F16 && std::fabs(x) < 6e-8f) { CHECK(std::fabs(y) < 6e-8f); continue; }
            const float tol = w == Wire::F16 ? (std::fabs(x) < 6.1e-5f ? 6e-8f : std::fabs(x) * 4.9e-4f)
                                             : std::fabs(x) * 3.9e-3f;
            CHECK(std::fabs(x - y) <= tol);
        }
    }
    // NaN stays NaN
    const float nan = std::nanf("");
    for (Wire w : {Wire::F16, Wire::BF16}) {
        uint8_t b[2];
        float y = 0;
        encode_rows(w, &nan, 1, b);
        decode_rows(w, b, 1, &y);
        CHECK(std::isnan(y));
    }
    // random values: round to nearest within half an ulp
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> v(100000), r(v.size());
    for (float& x : v) x = nd(rng);
    std::vector<uint8_t> b(v.size() * 2);
    encode_rows(Wire::F16, v.data(), v.size(), b.data());
    decode_rows(Wire::F16, b.data(), v.size(), r.data());
    double worst = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (std::fabs(v[i]) > 1e-4f) worst = std::max(worst, (double) std::fabs(v[i] - r[i]) / std::fabs(v[i]));
    CHECK(worst <= 4.9e-4);
    encode_rows(Wire::BF16, v.data(), v.size(), b.data());
    decode_rows(Wire::BF16, b.data(), v.size(), r.data());
    worst = 0;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] != 0.0f) worst = std::max(worst, (double) std::fabs(v[i] - r[i]) / std::fabs(v[i]));
    CHECK(worst <= 3.91e-3);
    Wire w;
    CHECK(parse_wire("f16", w) && w == Wire::F16 && parse_wire("bf16", w) && w == Wire::BF16 && !parse_wire("int8", w));
}

static void test_kv_and_addresses() {
    KV kv;
    kv.set("a", (int64_t) -42);
    kv.set("b", 0.1);
    kv.set("c", std::string("hello world=1"));
    kv.set("d", std::string("multi\nline"));
    const KV r = KV::decode(kv.encode());
    CHECK(r.i64("a") == -42 && std::fabs(r.f64("b") - 0.1) < 1e-15 && r.str("c") == "hello world=1");
    CHECK(r.str("d") == "multi line" && r.i64("missing", 7) == 7 && r.i64("c", 3) == 3);
    std::string h;
    int p = 0;
    CHECK(split_host_port("10.0.0.5:7701", 1, h, p) && h == "10.0.0.5" && p == 7701);
    CHECK(split_host_port("jenlap", 7701, h, p) && h == "jenlap" && p == 7701);
    CHECK(split_host_port(" laptop.local:8000 ", 1, h, p) && h == "laptop.local" && p == 8000);
    CHECK(split_host_port("[fe80::1]:7702", 1, h, p) && h == "fe80::1" && p == 7702);
    CHECK(!split_host_port("host:", 1, h, p) && !split_host_port("host:99999", 1, h, p) && !split_host_port("", 1, h, p));
    std::vector<int64_t> at;
    CHECK(parse_split("24", at) && at.size() == 1 && at[0] == 24);
    CHECK(parse_split("16,32", at) && at.size() == 2 && !parse_split("32,16", at) && !parse_split("a", at));
}

static void test_frames() {
    Listener l;
    std::string err;
    CHECK(l.open("127.0.0.1", 0, err));
    const int port = l.port();
    CHECK(port > 0);
    std::vector<float> big(3 << 20);   // 12 MB of rows
    for (size_t i = 0; i < big.size(); ++i) big[i] = (float) i * 0.5f;
    std::thread server([&] {
        std::string e;
        Socket s = l.accept(10.0, e);
        Channel c(std::move(s));
        Msg t;
        std::vector<uint8_t> p;
        // echo: a frame back with the same payload, as rows
        for (int i = 0; i < 3; ++i) {
            if (!c.recv(t, p, e)) return;
            if (t == Msg::Commit) { c.send(Msg::Ack, e); continue; }
            c.send(Msg::PrefillRows, p, e);
        }
        if (!c.recv(t, p, e)) return;            // a commit that fails
        KV no;
        no.set("message", std::string("no such window"));
        c.send(Msg::Err, no, e);
    });
    Channel c(connect_to("127.0.0.1", port, 5.0, err));
    CHECK(c.valid());
    c.socket().set_timeout(10.0);
    Packer pk;
    pk.put<int64_t>(5);
    pk.put<uint64_t>(77);
    CHECK(c.send(Msg::Prefill, pk.b.data(), pk.b.size(), big.data(), big.size() * 4, err));
    Msg t;
    std::vector<uint8_t> p;
    CHECK(c.recv(t, p, err) && t == Msg::PrefillRows && p.size() == pk.b.size() + big.size() * 4);
    Unpacker u(p.data(), p.size());
    int64_t a = 0;
    uint64_t b = 0;
    CHECK(u.get(a) && u.get(b) && a == 5 && b == 77 && u.left() == big.size() * 4);
    CHECK(std::memcmp(u.here(), big.data(), big.size() * 4) == 0);
    // a lazily acknowledged commit, then a frame read straight into a buffer
    Packer cm;
    cm.put<int32_t>(3);
    CHECK(c.send(Msg::Commit, cm.b, err));
    ++c.pending;
    CHECK(c.send(Msg::Verify, pk.b.data(), pk.b.size(), big.data(), 4096, err));
    CHECK(c.drain(err) && c.pending == 0);
    std::vector<uint8_t> prefix;
    std::vector<float> rows(1024);
    CHECK(c.recv_into(t, prefix, pk.b.size(), rows.data(), 4096, err) && t == Msg::PrefillRows);
    CHECK(prefix.size() == pk.b.size() && std::memcmp(rows.data(), big.data(), 4096) == 0);
    // an Err surfaces through drain with the worker's text
    CHECK(c.send(Msg::Commit, cm.b, err));
    ++c.pending;
    CHECK(!c.drain(err) && err.find("no such window") != std::string::npos);
    server.join();
    // a stranger: not our magic
    std::thread bad([&] {
        std::string e;
        Socket s = l.accept(10.0, e);
        const char junk[64] = "GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        s.send_all(junk, sizeof junk, e);
    });
    Channel c2(connect_to("127.0.0.1", port, 5.0, err));
    c2.socket().set_timeout(5.0);
    CHECK(!c2.recv(t, p, err) && err.find("magic") != std::string::npos);
    bad.join();
    // nobody listening: a clear error, quickly
    Listener tmp;
    CHECK(tmp.open("127.0.0.1", 0, err));
    const int dead = tmp.port();
    tmp.close();
    Socket none = connect_to("127.0.0.1", dead, 2.0, err);
    CHECK(!none.valid() && !err.empty());
}

// the coordinator's half of the handshake, as PoolLink does it (link.cpp needs CUDA, so it is spelled out here)
static bool coordinator_hello(Channel& c, const std::string& secret, KV& hello, std::string& err) {
    Msg t;
    std::vector<uint8_t> p;
    if (!c.recv(t, p, err) ) return false;
    if (t != Msg::Hello) { err = unexpected(t, Msg::Hello, p); return false; }
    hello = KV::decode(p);
    const std::string wn = hello.str("nonce"), cn = random_nonce();
    const auto mac = hmac_sha256(secret, wn.data(), wn.size());
    KV a;
    a.set("mac", to_hex(mac.data(), mac.size()));
    a.set("nonce", cn);
    if (!c.send(Msg::Auth, a, err) || !c.recv(t, p, err)) return false;
    if (t != Msg::AuthOk) { err = unexpected(t, Msg::AuthOk, p); return false; }
    const auto want = hmac_sha256(secret, cn.data(), cn.size());
    if (!equal_ct(KV::decode(p).str("mac"), to_hex(want.data(), want.size()))) { err = "worker secret"; return false; }
    KV cfg;
    cfg.set("lb", (int64_t) 24);
    cfg.set("le", (int64_t) 48);
    return c.send(Msg::Config, cfg, err);
}

static void test_handshake() {
    Listener l;
    std::string err;
    CHECK(l.open("127.0.0.1", 0, err));
    const int port = l.port();
    std::atomic<bool> stop{false};
    std::vector<std::string> logs;
    std::mutex log_mu;
    Channel served;
    KV cfg;
    bool accepted = false;
    std::thread worker([&] {
        accepted = accept_coordinator(
            l, "s3cret", [] { KV h; h.set("proto", (int64_t) kProtocol); h.set("name", std::string("w")); return h; },
            served, cfg, [&](const std::string& m) { std::lock_guard<std::mutex> g(log_mu); logs.push_back(m); }, &stop);
    });
    // a coordinator with the wrong secret is refused (and the worker keeps waiting)
    {
        Channel c(connect_to("127.0.0.1", port, 5.0, err));
        c.socket().set_timeout(10.0);
        KV hello;
        CHECK(!coordinator_hello(c, "wrong", hello, err));
        CHECK(err.find("secret is different") != std::string::npos);
    }
    // the right one gets through, and its CONFIG arrives
    Channel c(connect_to("127.0.0.1", port, 5.0, err));
    c.socket().set_timeout(10.0);
    KV hello;
    CHECK(coordinator_hello(c, "s3cret", hello, err));
    worker.join();
    CHECK(accepted && cfg.i64("lb") == 24 && cfg.i64("le") == 48 && hello.str("name") == "w");
    CHECK(hello.str("nonce").size() == 32);
    {
        std::lock_guard<std::mutex> g(log_mu);
        bool refused = false;
        for (const std::string& m : logs) refused = refused || m.find("wrong pool secret") != std::string::npos;
        CHECK(refused);
    }
    // while it serves, another coordinator hears "busy"
    {
        BusyResponder busy(l, served.peer());
        Channel c3(connect_to("127.0.0.1", port, 5.0, err));
        c3.socket().set_timeout(10.0);
        Msg t;
        std::vector<uint8_t> p;
        CHECK(c3.recv(t, p, err) && t == Msg::Err && KV::decode(p).i64("busy") == 1);
    }
    // stop ends a wait
    stop.store(true);
    Channel none;
    KV cfg2;
    CHECK(!accept_coordinator(l, "x", [] { return KV(); }, none, cfg2, [](const std::string&) {}, &stop));
}

static void test_split() {
    SplitModel m;
    m.n_layers = 48;
    for (int r = 0; r < 12288; ++r) m.profile.push_back({(int32_t) (r % 48), (int32_t) (r / 48)});
    m.slot_bytes = [](int64_t) { return (int64_t) 1 << 20; };              // 1 MiB per cached expert
    m.layer_arena_bytes.assign(48, (int64_t) 1 << 30);                     // 1 GiB of experts per layer
    m.session_bytes = [](int64_t lb, int64_t le) { return (le - lb) * ((int64_t) 8 << 20); };
    NodeCap a{"desktop", (int64_t) 4 << 30, (int64_t) 40 << 30, 0.5};
    NodeCap b{"laptop", (int64_t) 3 << 30, (int64_t) 20 << 30, 0.6};
    const SplitPlan p = auto_split(m, {a, b});
    CHECK(p.ok && p.at.size() == 1);
    CHECK(p.at[0] >= 28 && p.at[0] <= 40);                  // the laptop's RAM holds at most 20 of the 48 layers
    CHECK(p.node_arena[1] <= b.ram_room && p.node_arena[0] <= a.ram_room);
    CHECK(p.held > 0 && p.mass > 0 && p.mass <= 1.0);
    // a manual split the laptop's RAM cannot hold
    const SplitPlan bad = evaluate_split(m, {a, b}, {10});
    CHECK(!bad.ok && bad.why.find("RAM") != std::string::npos);
    // two PCs whose RAM together cannot hold the model: no placement
    NodeCap tiny{"tiny", (int64_t) 4 << 30, (int64_t) 10 << 30, 0.5};
    const SplitPlan none = auto_split(m, {tiny, tiny});
    CHECK(!none.ok && !none.why.empty());
    // three nodes: every boundary rising, every range non-empty
    const SplitPlan three = auto_split(m, {a, b, b});
    CHECK(three.ok && three.at.size() == 2 && three.at[0] >= 2 && three.at[1] > three.at[0] && three.at[1] < 48);
    // more VRAM -> more cached experts predicted
    NodeCap big_b{"laptop+", (int64_t) 12 << 30, (int64_t) 20 << 30, 0.6};
    const SplitPlan more = auto_split(m, {a, big_b});
    CHECK(more.ok && more.held >= p.held);
    // the coordinator keeps layers 0 and 1
    const SplitPlan one = evaluate_split(m, {a, b}, {1});
    CHECK(!one.ok);
}

int main() {
    if (!net_init()) { std::fprintf(stderr, "no sockets\n"); return 1; }
    test_hashes();
    test_wire();
    test_kv_and_addresses();
    test_frames();
    test_handshake();
    test_split();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("strata-pool-test: all checks passed\n");
    return 0;
}
