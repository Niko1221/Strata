// src/pool/worker.cpp - the pool: a worker meets its coordinator.
#include "strata/pool/worker.hpp"

#include <chrono>
#include <vector>

namespace strata::pool {

bool accept_coordinator(Listener& l, const std::string& secret, const std::function<KV()>& hello, Channel& out,
                        KV& cfg, const std::function<void(const std::string&)>& log, const std::atomic<bool>* stop) {
    for (;;) {
        if (stop != nullptr && stop->load()) return false;
        std::string e;
        Socket so = l.accept(stop != nullptr ? 0.5 : -1.0, e);
        if (!so.valid()) {
            if (!e.empty()) {
                log(e);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            continue;
        }
        so.set_timeout(30.0);
        Channel ch(std::move(so));
        const std::string peer = ch.peer();
        const std::string nonce = random_nonce();
        KV h = hello();
        h.set("nonce", nonce);
        Msg t;
        std::vector<uint8_t> p;
        if (!ch.send(Msg::Hello, h, e) || !ch.recv(t, p, e, 1 << 20)) {
            log(peer + ": " + e);
            continue;
        }
        if (t != Msg::Auth) {
            log(peer + ": " + unexpected(t, Msg::Auth, p));
            continue;
        }
        const KV auth = KV::decode(p);
        const auto mac = hmac_sha256(secret, nonce.data(), nonce.size());
        if (!equal_ct(auth.str("mac"), to_hex(mac.data(), mac.size()))) {
            KV no;
            no.set("message", std::string("this worker's pool secret is different"));
            ch.send(Msg::Err, no, e);
            log("refused " + peer + ": wrong pool secret");
            continue;
        }
        const std::string cn = auth.str("nonce");
        const auto mine = hmac_sha256(secret, cn.data(), cn.size());
        KV ok;
        ok.set("mac", to_hex(mine.data(), mine.size()));
        if (!ch.send(Msg::AuthOk, ok, e) || !ch.recv(t, p, e, 1 << 20)) {
            log(peer + ": " + e);
            continue;
        }
        if (t != Msg::Config) {
            log(peer + ": " + unexpected(t, Msg::Config, p));
            continue;
        }
        cfg = KV::decode(p);
        ch.socket().set_timeout(0);
        out = std::move(ch);
        return true;
    }
}

BusyResponder::BusyResponder(Listener& l, std::string serving) {
    t_ = std::thread([this, &l, serving] {
        while (on_.load()) {
            std::string e;
            Socket so = l.accept(0.5, e);
            if (!so.valid()) continue;
            so.set_timeout(5.0);
            Channel c(std::move(so));
            KV no;
            no.set("busy", (int64_t) 1);
            no.set("message", "serving the coordinator at " + serving);
            c.send(Msg::Err, no, e);
        }
    });
}

BusyResponder::~BusyResponder() {
    on_.store(false);
    if (t_.joinable()) t_.join();
}

}  // namespace strata::pool
