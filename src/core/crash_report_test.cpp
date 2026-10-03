// crash_report_test.cpp - the B0 gate: a crashing engine must SAY so on stderr and still die by its signal.
// The preserved B0 death (bench/results/prefill-preempt 2026-10-03) left no stderr at all; the handler turns a
// silent SIGSEGV/SIGABRT into a "strata: FATAL ..." note plus a backtrace, then re-raises - the returncode keeps
// naming the signal for the harness.  POSIX only (the handler is a no-op elsewhere, like the feature).
#include "strata/core/crash_report.hpp"

#if defined(STRATA_CRASH_REPORT)

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

/// Runs `body` in a forked child with the handlers installed; returns {exited, status, stderr text}.
struct Crash {
    bool signaled = false;
    int sig = 0;
    std::string err;
};

static Crash crash_of(void (*body)()) {
    Crash out;
    int fds[2];
    if (pipe(fds) != 0) { std::fprintf(stderr, "FAIL: pipe\n"); std::exit(1); }
    const pid_t pid = fork();
    if (pid < 0) { std::fprintf(stderr, "FAIL: fork\n"); std::exit(1); }
    if (pid == 0) {
        ::close(fds[0]);
        dup2(fds[1], 2);
        strata::core::install_crash_report();
        body();
        _exit(0);                     // a body that survives means the test set it up wrong
    }
    ::close(fds[1]);
    char buf[4096];
    for (ssize_t n; (n = read(fds[0], buf, sizeof buf)) > 0;) out.err.append(buf, (size_t) n);
    ::close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    out.signaled = WIFSIGNALED(status);
    out.sig = out.signaled ? WTERMSIG(status) : 0;
    return out;
}

static void segv_body() {
    volatile int* null_p = nullptr;
    *null_p = 1;                      // deliberate SIGSEGV: the fault address must name page 0
}
static void abort_body() { std::abort(); }

int main() {
    const Crash segv = crash_of(segv_body);
    check(segv.signaled && segv.sig == SIGSEGV, "the process still dies by SIGSEGV (returncode keeps naming it)");
    check(segv.err.find("strata: FATAL SIGSEGV") != std::string::npos, "the stderr note names the signal");
    check(segv.err.find("fault address 0x") != std::string::npos, "the stderr note carries the fault address");
    check(segv.err.find("backtrace") != std::string::npos && segv.err.find("+0x") != std::string::npos,
          "the stderr note carries a backtrace");

    const Crash abrt = crash_of(abort_body);
    check(abrt.signaled && abrt.sig == SIGABRT, "abort() still ends in SIGABRT (no handler loop)");
    check(abrt.err.find("strata: FATAL SIGABRT") != std::string::npos, "abort() is reported the same way");

    // a clean exit must stay untouched: nothing is printed and the status is the program's own
    const pid_t pid = fork();
    if (pid == 0) { strata::core::install_crash_report(); _exit(7); }
    int status = 0;
    waitpid(pid, &status, 0);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 7, "a clean exit keeps its own status");
    std::printf("%d checks passed\n", checks);
}

#else  // no backtrace API (Windows): the feature is a no-op, so this test only proves it compiles

int main() {
    strata::core::install_crash_report();
    std::printf("0 checks passed\n");
}

#endif
