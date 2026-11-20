#pragma once

#include "protocol.hpp"
#include <windows.h>
#include <functional>
#include <thread>

// Owns the thread, not the input handle or protocol. Both must outlive it.
// The optional boundary callbacks let CPU pipe tests force the check/read race.
class Q38InputReader {
public:
    struct Hooks {
        std::function<void()> before_read;
        std::function<void(DWORD)> after_cancel;
    };

    Q38InputReader(HANDLE input, Q38Protocol& state, Hooks hooks = {})
        : input_(input), state_(state), hooks_(std::move(hooks)),
          reader_([this] { read(); }) {}
    Q38InputReader(const Q38InputReader&) = delete;
    Q38InputReader& operator=(const Q38InputReader&) = delete;
    ~Q38InputReader() { stop_and_join(); }

    void stop_and_join() noexcept {
        if (!reader_.joinable()) return;
        stopping_.store(true);
        // Cancellation only affects an outstanding read. Keep cancelling until
        // the thread actually exits: a read begun after ERROR_NOT_FOUND must
        // also be cancelled. The stop flag prevents any subsequent read.
        for (;;) {
            const DWORD waited = WaitForSingleObject(reader_.native_handle(), 0);
            if (waited == WAIT_OBJECT_0) break;
            if (waited != WAIT_TIMEOUT) std::terminate();
            const DWORD error = CancelSynchronousIo(reader_.native_handle())
                ? ERROR_SUCCESS : GetLastError();
            if (error != ERROR_SUCCESS && error != ERROR_NOT_FOUND) std::terminate();
            if (hooks_.after_cancel) hooks_.after_cancel(error);
            const DWORD done = WaitForSingleObject(reader_.native_handle(), 10);
            if (done == WAIT_OBJECT_0) break;
            if (done != WAIT_TIMEOUT) std::terminate();
        }
        reader_.join();
    }

private:
    void read() {
        std::string line;
        bool overflow = false;
        char buffer[4096];
        while (!stopping_.load()) {
            if (hooks_.before_read) hooks_.before_read();
            DWORD count = 0;
            if (!ReadFile(input_, buffer, sizeof(buffer), &count, nullptr) || !count) break;
            for (DWORD i = 0; i < count && !stopping_.load(); ++i) {
                const char c = buffer[i];
                if (c == '\n') {
                    if (!overflow) {
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        state_.submit(line);
                    } else state_.submit("INVALID_LINE_TOO_LONG");
                    line.clear();
                    overflow = false;
                } else if (!overflow) {
                    if (line.size() >= 4 * 1024 * 1024) { line.clear(); overflow = true; }
                    else line += c;
                }
            }
        }
        state_.end_input();
    }

    HANDLE input_;
    Q38Protocol& state_;
    Hooks hooks_;
    std::atomic<bool> stopping_{false};
    std::thread reader_;
};
