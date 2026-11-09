// Cooperative foreground fairness. Commands carry a short lease so losing the
// server cannot leave the engine parked indefinitely. No allocation or CUDA work
// happens here; the caller owns safe-point draining, STOP handling and sleeping.
#pragma once

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace strata::core {

class BackgroundControl {
public:
    bool ingest(const std::string& line, int64_t now_ms) {
        if (line.size() > 80 || now_ms < 0) return false;
        std::istringstream in(line);
        std::string verb, extra;
        int64_t lease = -1, delay = -1, wait = -1;
        if (!(in >> verb >> lease >> delay >> wait) || (in >> extra) || verb != "BACKGROUND" ||
            lease < 0 || lease > 5000 || delay < 0 || delay > 100 || wait < 0 || wait > 1 ||
            (lease == 0 && (delay != 0 || wait != 0)) ||
            now_ms > std::numeric_limits<int64_t>::max() - lease) return false;
        since_ = now_ms;
        until_ = now_ms + lease;
        delay_ = (int) delay;
        wait_ = wait != 0;
        return true;
    }

    bool active(int64_t now_ms) const { return now_ms >= since_ && now_ms < until_; }
    bool waiting(int64_t now_ms) const { return active(now_ms) && wait_; }
    int delay_ms(int64_t now_ms) const { return active(now_ms) ? delay_ : 0; }
    int remaining_ms(int64_t now_ms) const { return active(now_ms) ? (int) (until_ - now_ms) : 0; }

private:
    int64_t since_ = 0, until_ = 0;
    int delay_ = 0;
    bool wait_ = false;
};

}  // namespace strata::core
