// include/strata/pool/worker.hpp - the pool's worker side of the handshake (the engine's worker loop is in
// generate.cpp, beside the serve loop whose lambdas it shares).
#pragma once

#include "strata/pool/protocol.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace strata::pool {

/// Waits on `l` for a coordinator: sends it `hello()` with a fresh nonce, checks its proof of `secret`, proves it
/// back, and takes its CONFIG into `cfg`.  A peer that fails any step is logged through `log` and dropped, and the
/// wait goes on; `stop` (optional) ends the wait early (returns false).  The channel is left without a timeout (a
/// coordinator may stay quiet for hours between requests).
bool accept_coordinator(Listener& l, const std::string& secret, const std::function<KV()>& hello, Channel& out,
                        KV& cfg, const std::function<void(const std::string&)>& log,
                        const std::atomic<bool>* stop = nullptr);

/// While a worker serves one coordinator, any other that connects hears "busy" (and waits or gives up) instead of
/// hanging in the listen backlog.
class BusyResponder {
public:
    BusyResponder(Listener& l, std::string serving);
    ~BusyResponder();
    BusyResponder(const BusyResponder&) = delete;
    BusyResponder& operator=(const BusyResponder&) = delete;

private:
    std::atomic<bool> on_{true};
    std::thread t_;
};

}  // namespace strata::pool
