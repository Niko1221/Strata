// SPDX-License-Identifier: MIT
// Run each watchdog case in a fresh process; the production helper calls _Exit.
#include <cstdio>

#if defined(_WIN32) || !defined(__unix__)
int main() {
    std::fputs("SKIP: bounded child-process test requires POSIX\n", stdout);
    return 77;
}
#else
#include "strata/core/device_lost_exit.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono_literals;

static int child_case(const char* name) {
    using strata::core::arm_device_lost_exit;
    if (std::strcmp(name, "disabled") == 0) {
        if (setenv("STRATA_LOST_EXIT_S", "0", 1) != 0) return 91;
        arm_device_lost_exit("FIRST_REASON");
        std::this_thread::sleep_for(1400ms);
        return 0;
    }
    if (std::strcmp(name, "shorten") == 0) {
        // A large initial deadline separates shortening from scheduling noise.
        arm_device_lost_exit("FIRST_REASON", 20);
        std::this_thread::sleep_for(200ms);
        arm_device_lost_exit("LATER_REASON", 1);
    } else if (std::strcmp(name, "single") == 0 || std::strcmp(name, "rearm") == 0) {
        arm_device_lost_exit("FIRST_REASON", 1);
    } else {
        return 92;
    }
    for (;;) {
        std::this_thread::sleep_for(50ms);
        if (std::strcmp(name, "rearm") == 0) arm_device_lost_exit("LATER_REASON", 1);
    }
}

struct Outcome {
    int status = 0;
    bool timeout = false;
    bool skipped = false;
    bool io_error = false;
    std::string text;
};

static Outcome run_child(const char* executable, const char* name) {
    Outcome result;
    int channel[2];
    if (pipe(channel) != 0) { result.skipped = true; return result; }
    const pid_t pid = fork();
    if (pid < 0) {
        close(channel[0]); close(channel[1]);
        result.skipped = true;
        return result;
    }
    if (pid == 0) {
        close(channel[0]);
        if (dup2(channel[1], STDOUT_FILENO) < 0 || dup2(channel[1], STDERR_FILENO) < 0)
            std::_Exit(93);
        close(channel[1]);
        execl(executable, executable, "--child", name, static_cast<char*>(nullptr));
        std::_Exit(94);
    }
    close(channel[1]);
    const int flags = fcntl(channel[0], F_GETFL);
    if (flags < 0 || fcntl(channel[0], F_SETFL, flags | O_NONBLOCK) < 0) result.io_error = true;
    const auto deadline = std::chrono::steady_clock::now() + 6s;
    auto drain = [&] {
        char bytes[512];
        for (;;) {
            const ssize_t n = read(channel[0], bytes, sizeof(bytes));
            if (n > 0) {
                result.text.append(bytes, static_cast<size_t>(n));
                if (result.text.size() > 8192) { result.io_error = true; return; }
            } else {
                if (n < 0 && errno == EINTR) continue;
                if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) result.io_error = true;
                return;
            }
        }
    };
    bool reaped = false;
    while (!result.io_error) {
        drain();
        const pid_t finished = waitpid(pid, &result.status, WNOHANG);
        if (finished == pid) { reaped = true; break; }
        if (finished < 0 && errno != EINTR) { result.io_error = true; break; }
        if (std::chrono::steady_clock::now() >= deadline) { result.timeout = true; break; }
        std::this_thread::sleep_for(10ms);
    }
    if (!reaped) {
        kill(pid, SIGKILL);
        while (waitpid(pid, &result.status, 0) < 0 && errno == EINTR) {}
    }
    if (!result.io_error) drain();
    close(channel[0]);
    return result;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) return child_case(argv[2]);
    const char* cases[] = {"single", "rearm", "shorten", "disabled"};
    int failures = 0;
    for (const char* name : cases) {
        const Outcome result = run_child(argv[0], name);
        if (result.skipped) {
            std::fputs("SKIP: child process could not be created\n", stdout);
            return 77;
        }
        const bool disabled = std::strcmp(name, "disabled") == 0;
        bool ok = !result.timeout && !result.io_error && WIFEXITED(result.status) &&
                  WEXITSTATUS(result.status) == (disabled ? 0 : 3);
        if (disabled) {
            ok = ok && result.text.empty();
        } else {
            ok = ok && result.text.find("strata: fatal GPU error (FIRST_REASON)") != std::string::npos &&
                 result.text.find("ERR fatal GPU error: FIRST_REASON") != std::string::npos &&
                 result.text.find("LATER_REASON") == std::string::npos;
        }
        std::printf("%s %s (timeout=%d)\n", ok ? "PASS" : "FAIL", name, result.timeout);
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "status=%d io_error=%d output=%s\n", result.status,
                         result.io_error, result.text.c_str());
        }
    }
    return failures ? 1 : 0;
}
#endif
