// crash_report.hpp - make a crash SAY something on stderr.
//
// **WHY THIS EXISTS.** bench/results/prefill-preempt (2026-10-03) lost an engine to an unexplained death: the
// harness saw stdout EOF, the engine's own stderr ended mid-decode with no error, and neither the kernel log nor
// a core dump could be recovered - so the death was undiagnosable after the fact ("B0").  Every deliberate error
// path in the engine prints; what a signal leaves behind is nothing.  These handlers convert a silent
// SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT into a stderr note (signal, si_code, fault address, backtrace of the
// faulting thread) and then restore the default disposition and re-raise, so the process still dies by the signal
// it hit and the returncode keeps naming it.
//
// The handler only calls async-signal-safe things: write(2), a hand-rolled hex formatter, and glibc's
// backtrace/backtrace_symbols_fd (documented async-signal-safe).  It runs on its own sigaltstack, so a stack
// overflow still reaches it.  Diagnostics, not recovery: nothing here changes what the engine computes.
#pragma once

#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#if defined(__GLIBC__)
#include <execinfo.h>
#include <unistd.h>
#define STRATA_CRASH_REPORT 1
#endif

namespace strata::core {
namespace crash {

#ifdef STRATA_CRASH_REPORT

inline void write_all(const char* s) {           // async-signal-safe: a plain write(2) loop
    size_t n = std::strlen(s);
    while (n > 0) {
        const ssize_t w = ::write(2, s, n);
        if (w <= 0) return;
        s += (size_t) w;
        n -= (size_t) w;
    }
}

inline void write_hex(uint64_t v) {
    char buf[19] = "0x";
    for (int i = 15; i >= 0; --i) {
        const int d = (int) (v & 0xF);
        buf[2 + i] = (char) (d < 10 ? '0' + d : 'a' + d - 10);
        v >>= 4;
    }
    buf[18] = '\0';
    write_all(buf);
}

inline void handler(int sig, siginfo_t* info, void*) {
    static const char* names[] = {"?", "?", "?", "?", "SIGILL", "SIGTRAP", "SIGABRT", "SIGBUS", "SIGFPE",
                                  "SIGKILL", "SIGUSR1", "SIGSEGV", "SIGUSR2", "SIGPIPE", "SIGALRM",
                                  "SIGTERM"};
    write_all("\nstrata: FATAL ");
    write_all(sig >= 0 && sig < 16 ? names[sig] : "signal");
    write_all(" (");
    write_hex((uint64_t) sig);
    write_all("), si_code ");
    write_hex((uint64_t) (info ? info->si_code : 0));
    write_all(", fault address ");
    write_hex(info ? (uint64_t) (uintptr_t) info->si_addr : 0);
    write_all(", backtrace of the faulting thread:\n");
    void* frames[64];
    const int n = ::backtrace(frames, 64);
    ::backtrace_symbols_fd(frames, n, 2);       // async-signal-safe per glibc
    ::raise(sig);                               // SA_RESETHAND restored SIG_DFL on entry: this one kills
}

#endif  // STRATA_CRASH_REPORT

}  // namespace crash

/// Install the handlers once, at the top of main.  A no-op where the backtrace API does not exist (Windows keeps
/// its own crash UI).  Never changes what the engine computes; a clean exit or a QUIT path is untouched.
inline void install_crash_report() {
#ifdef STRATA_CRASH_REPORT
    static bool done = false;
    if (done) return;
    done = true;
    stack_t ss{};
    ss.ss_sp = malloc(64 * 1024);               // the handler's own stack: a stack overflow still reports
    if (ss.ss_sp != nullptr) {
        ss.ss_size = 64 * 1024;
        ss.ss_flags = 0;
        sigaltstack(&ss, nullptr);
    }
    struct sigaction sa{};
    sa.sa_sigaction = crash::handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;   // RESETHAND: a second fault dies by default, no loop
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT})
        sigaction(sig, &sa, nullptr);
#endif
}

}  // namespace strata::core
